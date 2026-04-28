"""
skill_vision.py — Multimodal screen vision via moondream2
==========================================================
Lets Iris answer questions about what's on screen.

Triggers
--------
"what's on my screen"  / "what am I looking at"
"describe my screen"   / "what is this"
"read this"            / "what does this say"
"look at my screen"    / "can you see my screen"

Setup
-----
    ollama pull moondream

Screenshot method (KDE Wayland)
-------------------------------
Uses `grim` (Wayland native) → fallback `spectacle -b -n -o` → fallback
PIL ImageGrab (XWayland).  Works headless-ish: if no display found, returns
a helpful error message instead of crashing.

Priority 18 — checked before LLM fallback, after system/app intents.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

log = logging.getLogger(__name__)

METADATA = {
    "name":        "Vision",
    "version":     "1.0",
    "description": "Screen awareness via moondream2 — describe screen, read text, answer visual questions",
    "author":      "iris",
}

# Model used for vision — pull with: ollama pull moondream
VISION_MODEL = os.getenv("IRIS_VISION_MODEL", "moondream")

# Max pixels on the longest edge before we downsample (saves VRAM + speed)
MAX_EDGE_PX = int(os.getenv("IRIS_VISION_MAX_PX", "1280"))

# ── trigger phrases ────────────────────────────────────────────────────────
_VISION_TRIGGERS = [
    "what's on my screen", "whats on my screen",
    "what is on my screen", "what am i looking at",
    "describe my screen", "describe the screen",
    "what is this", "what's this",
    "read this", "read the screen", "read this text",
    "what does this say", "what does it say",
    "look at my screen", "can you see my screen",
    "what do you see", "analyse my screen", "analyze my screen",
    "summarize my screen", "summarise my screen",
    "what's open", "whats open", "what is open on my screen",
]

_VISION_RE = re.compile(
    r"\b(screen|display|monitor|window|look|see|read|describe|analyse|analyze|"
    r"summarize|summarise|this text|what.{0,10}open)\b",
    re.IGNORECASE,
)


def _match_vision(norm: str) -> bool:
    if any(t in norm for t in _VISION_TRIGGERS):
        return True
    # Broader: "what's open", "tell me what you see", etc.
    return bool(_VISION_RE.search(norm)) and any(
        w in norm for w in ["screen", "monitor", "display", "see", "look", "open"]
    )


# ── screenshot helpers ─────────────────────────────────────────────────────

def _screenshot_grim() -> bytes | None:
    """Wayland-native via grim."""
    if not shutil.which("grim"):
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        result = subprocess.run(
            ["grim", path],
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            data = Path(path).read_bytes()
            os.unlink(path)
            return data
        os.unlink(path)
    except Exception as e:
        log.debug("grim screenshot failed: %s", e)
    return None


def _screenshot_spectacle() -> bytes | None:
    """KDE Spectacle fallback."""
    if not shutil.which("spectacle"):
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        result = subprocess.run(
            ["spectacle", "-b", "-n", "-f", "-o", path],
            timeout=8,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0 and Path(path).exists():
            data = Path(path).read_bytes()
            os.unlink(path)
            return data
        if Path(path).exists():
            os.unlink(path)
    except Exception as e:
        log.debug("spectacle screenshot failed: %s", e)
    return None


def _screenshot_pil() -> bytes | None:
    """PIL ImageGrab — works on X11 / XWayland."""
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        log.debug("PIL screenshot failed: %s", e)
    return None


def _screenshot_import() -> bytes | None:
    """ImageMagick `import` — last resort X11 fallback."""
    if not shutil.which("import"):
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        result = subprocess.run(
            ["import", "-window", "root", path],
            timeout=6,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0 and Path(path).exists():
            data = Path(path).read_bytes()
            os.unlink(path)
            return data
        if Path(path).exists():
            os.unlink(path)
    except Exception as e:
        log.debug("ImageMagick screenshot failed: %s", e)
    return None


def take_screenshot() -> bytes | None:
    """Try all screenshot methods in order; return PNG bytes or None."""
    for method in (_screenshot_grim, _screenshot_spectacle, _screenshot_pil, _screenshot_import):
        data = method()
        if data:
            log.info("Screenshot captured via %s  (%d bytes)", method.__name__, len(data))
            return data
    return None


def _resize_image(png_bytes: bytes, max_edge: int = MAX_EDGE_PX) -> bytes:
    """Downsample if the image is very large — saves VRAM + inference time."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))
        w, h = img.size
        if max(w, h) <= max_edge:
            return png_bytes
        scale = max_edge / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        log.info("Image resized: %dx%d → %dx%d", w, h, new_w, new_h)
        return buf.getvalue()
    except Exception as e:
        log.debug("Image resize failed, using original: %s", e)
        return png_bytes


# ── vision query ───────────────────────────────────────────────────────────

def _build_vision_prompt(command: str) -> str:
    """Turn the user's voice command into a crisp moondream prompt."""
    cmd = command.lower().strip()

    # Explicit read/OCR intent
    if any(w in cmd for w in ["read", "what does it say", "what does this say", "text"]):
        return "Read all text visible on this screen. Be complete and accurate."

    # Asking what's open / what app
    if any(w in cmd for w in ["what's open", "whats open", "what is open", "what app"]):
        return (
            "What application or program is open on this screen? "
            "Name the app and briefly describe what the user is doing."
        )

    # Generic describe
    return (
        "Describe what is on this screen concisely in 2-3 sentences. "
        "Mention the main application, what the user appears to be working on, "
        "and any important visible content. Be specific and direct."
    )


def query_vision(png_bytes: bytes, command: str) -> str:
    """Send screenshot to moondream via ollama and return the reply."""
    import ollama

    prompt  = _build_vision_prompt(command)
    b64_img = base64.b64encode(png_bytes).decode("utf-8")

    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role":    "user",
                "content": prompt,
                "images":  [b64_img],
            }],
        )
        reply = (response.get("message") or {}).get("content", "").strip()
        return reply or "I couldn't read the screen clearly."
    except Exception as e:
        log.warning("moondream query failed: %s", e)
        # Give a helpful install message
        if "model" in str(e).lower() or "not found" in str(e).lower():
            return (
                f"The vision model isn't available. "
                f"Run: ollama pull {VISION_MODEL}"
            )
        return f"Vision query failed: {str(e)[:80]}"


# ── handler ────────────────────────────────────────────────────────────────

def _handle_vision(command: str, ctx: dict) -> str | None:
    speak = ctx["speak"]

    speak("Taking a screenshot, one moment.")

    png = take_screenshot()
    if not png:
        return (
            "I couldn't capture your screen. Make sure grim is installed on Wayland "
            "(sudo pacman -S grim) or spectacle for KDE."
        )

    # Resize before sending — 1280px longest edge is plenty for moondream
    png = _resize_image(png, max_edge=MAX_EDGE_PX)

    # Run inference (can take 1-3s on RTX 3050)
    reply = query_vision(png, command)
    return reply


INTENTS = [
    {
        "name":     "vision_screen",
        "priority": 18,
        "match":    _match_vision,
        "handle":   _handle_vision,
    },
]
