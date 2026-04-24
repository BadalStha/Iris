"""
skill_volume.py — Volume and brightness control
Priority 25.
"""

import re
import shutil
import subprocess

METADATA = {
    "name":        "Volume & Brightness",
    "version":     "1.0",
    "description": "Adjust system volume and screen brightness",
    "author":      "iris",
}

_VOL_WORDS   = ["volume", "sound", "louder", "quieter"]
_BRIGHT_WORDS = ["brightness", "brighter", "dimmer", "screen"]


def _norm(text):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split())


# ── volume parsing ──────────────────────────────────────────────────────────

def _parse_volume(norm):
    if re.search(r"\bmute\b", norm):
        return {"mode": "mute"}

    for pat in [
        r"(?:set\s+)?(?:the\s+)?(?:volume|sound)(?:\s+level)?\s*(?:to|at)\s*(\d{1,3})\s*%?",
        r"(?:set\s+)?(?:the\s+)?(?:volume|sound)\s*(\d{1,3})\s*%",
        r"(?:decrease|increase|raise|lower|turn\s+up|turn\s+down)\s+(?:the\s+)?(?:volume|sound)\s*(?:to|at)\s*(\d{1,3})\s*%?",
    ]:
        m = re.search(pat, norm)
        if m:
            lvl = max(0, min(100, int(m.group(1))))
            return {"mode": "mute"} if lvl == 0 else {"mode": "absolute", "level": lvl}

    for pat, direction in [
        (r"\b(?:volume|sound)\s*(?:up|higher|increase|raise|louder)\b", "up"),
        (r"\b(?:volume|sound)\s*(?:down|lower|decrease|reduce|quieter)\b", "down"),
        (r"\b(?:turn\s+up|make\s+it\s+louder|increase|raise)\s+(?:the\s+)?(?:volume|sound)\b", "up"),
        (r"\b(?:turn\s+down|make\s+it\s+quieter|decrease|lower|reduce)\s+(?:the\s+)?(?:volume|sound)\b", "down"),
    ]:
        if re.search(pat, norm):
            return {"mode": "relative", "direction": direction, "amount": 10}

    if any(w in norm for w in _VOL_WORDS):
        up_words = ["up", "higher", "louder", "increase", "raise"]
        direction = "up" if any(w in norm for w in up_words) else "down"
        return {"mode": "relative", "direction": direction, "amount": 10}

    return None


# ── brightness parsing ──────────────────────────────────────────────────────

def _parse_brightness(norm):
    for pat in [
        r"(?:set\s+)?(?:the\s+)?brightness(?:\s+level)?\s*(?:to|at)\s*(\d{1,3})\s*%?",
        r"(?:set\s+)?(?:the\s+)?screen\s+brightness\s*(?:to|at)\s*(\d{1,3})\s*%?",
    ]:
        m = re.search(pat, norm)
        if m:
            return {"mode": "absolute", "level": max(0, min(100, int(m.group(1))))}

    for pat, direction in [
        (r"\bbrightness\s*(?:up|higher|increase|raise|brighter)\b", "up"),
        (r"\bbrightness\s*(?:down|lower|decrease|reduce|dimmer)\b", "down"),
        (r"\b(?:turn\s+up|make\s+it\s+brighter|increase|raise)\s+(?:the\s+)?brightness\b", "up"),
        (r"\b(?:turn\s+down|make\s+it\s+dimmer|decrease|lower|reduce)\s+(?:the\s+)?brightness\b", "down"),
    ]:
        if re.search(pat, norm):
            return {"mode": "relative", "direction": direction, "amount": 10}

    if any(w in norm for w in _BRIGHT_WORDS):
        up_words = ["up", "higher", "brighter", "increase", "raise"]
        direction = "up" if any(w in norm for w in up_words) else "down"
        return {"mode": "relative", "direction": direction, "amount": 10}

    return None


# ── pactl / brightnessctl helpers ───────────────────────────────────────────

def _set_vol_relative(direction, amount=10):
    sign = "+" if direction == "up" else "-"
    subprocess.run(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{amount}%"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return f"Volume {'up' if direction == 'up' else 'down'}."


def _set_vol_absolute(level):
    if level == 0:
        subprocess.run(
            ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return "Muted."
    subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"Volume set to {level} percent."


def _set_bright_relative(direction, amount=10):
    if not shutil.which("brightnessctl"):
        return "brightnessctl is not installed. Run: yay -S brightnessctl"
    sign = "+" if direction == "up" else "-"
    subprocess.run(
        ["brightnessctl", "set", f"{amount}%{sign}"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return f"Brightness {'up' if direction == 'up' else 'down'}."


def _set_bright_absolute(level):
    if not shutil.which("brightnessctl"):
        return "brightnessctl is not installed. Run: yay -S brightnessctl"
    subprocess.run(
        ["brightnessctl", "set", f"{level}%"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return f"Brightness set to {level} percent."


# ── match / handle ──────────────────────────────────────────────────────────

def _match_volume(norm):
    return bool(_parse_volume(norm))


def _match_brightness(norm):
    return bool(_parse_brightness(norm))


def _handle_volume(command, ctx):
    req = _parse_volume(_norm(command))
    if not req:
        return "I didn't understand that volume command."
    try:
        if req["mode"] == "mute":
            return _set_vol_absolute(0)
        elif req["mode"] == "absolute":
            return _set_vol_absolute(req["level"])
        else:
            return _set_vol_relative(req["direction"], req.get("amount", 10))
    except Exception as exc:
        return f"I couldn't adjust the volume: {str(exc)[:60]}"


def _handle_brightness(command, ctx):
    req = _parse_brightness(_norm(command))
    if not req:
        return "I didn't understand that brightness command."
    try:
        if req["mode"] == "absolute":
            return _set_bright_absolute(req["level"])
        else:
            return _set_bright_relative(req["direction"], req.get("amount", 10))
    except Exception as exc:
        return f"I couldn't adjust the brightness: {str(exc)[:60]}"


INTENTS = [
    {"name": "volume",     "priority": 25, "match": _match_volume,     "handle": _handle_volume},
    {"name": "brightness", "priority": 25, "match": _match_brightness, "handle": _handle_brightness},
]
