"""
skill_automation.py — System desktop automation
=================================================
Extends KWin scripting with:
  • Desktop notifications via notify-send / KDE D-Bus
  • Active window detection — "what am I working on?"
  • Monitor / display switching via kscreen-doctor
  • Quick system actions: lock screen, suspend, screenshot save

Trigger examples
----------------
"notify me about the meeting"
"send a notification: standup in 5 minutes"
"what am I working on"          / "what window is focused"
"switch to monitor 2"           / "turn off display 1"
"lock the screen"               / "lock screen"
"save a screenshot"             / "take a screenshot and save it"

Priority 22 — after vision (18), before apps (25).

Install
-------
    sudo pacman -S libnotify xdotool
    # kscreen-doctor is bundled with KDE plasma-workspace
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

METADATA = {
    "name":        "Automation",
    "version":     "1.0",
    "description": "Desktop notifications, active window detection, monitor switching",
    "author":      "iris",
}

# Where to save screenshots when the user asks
SCREENSHOT_DIR = Path.home() / "Pictures" / "iris-screenshots"


# ══════════════════════════════════════════════════════════════════════════
# NOTIFY-SEND  (desktop notification)
# ══════════════════════════════════════════════════════════════════════════

def _send_notification(title: str, body: str = "", urgency: str = "normal",
                       icon: str = "dialog-information") -> bool:
    """
    Send a desktop notification.
    Tries notify-send first, then KDE D-Bus fallback.
    """
    # 1. notify-send (libnotify)
    if shutil.which("notify-send"):
        try:
            subprocess.run(
                ["notify-send", "--urgency", urgency, "--icon", icon, title, body],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return True
        except Exception as e:
            log.debug("notify-send failed: %s", e)

    # 2. KDE D-Bus fallback (works without libnotify)
    qdbus = shutil.which("qdbus6") or shutil.which("qdbus-qt6") or shutil.which("qdbus")
    if qdbus:
        try:
            script = (
                f'Notify("{title}", "{body}")'
                if body else f'Notify("{title}")'
            )
            subprocess.run(
                [
                    qdbus,
                    "org.freedesktop.Notifications",
                    "/org/freedesktop/Notifications",
                    "org.freedesktop.Notifications.Notify",
                    "Iris", "0", icon, title, body,
                    "[]", "{}", "5000",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return True
        except Exception as e:
            log.debug("D-Bus notify failed: %s", e)

    return False


def _parse_notification(command: str) -> tuple[str, str]:
    """Extract title and body from commands like 'notify me: standup in 5 min'."""
    cmd = command.lower()
    # Pattern: "notify/notification/remind: <content>"
    m = re.search(
        r"(?:notify(?:\s+me)?|send\s+(?:a\s+)?notification|notification)\s*[:\-]?\s*(.+)",
        cmd, re.IGNORECASE,
    )
    body = m.group(1).strip().rstrip(".,?!") if m else command.strip()
    return "Iris", body


def _match_notify(norm: str) -> bool:
    return any(p in norm for p in [
        "notify me", "send a notification", "send notification",
        "desktop notification", "pop up", "popup",
    ])


def _handle_notify(command: str, ctx: dict) -> str:
    title, body = _parse_notification(command)
    ok = _send_notification(title, body)
    if ok:
        return f"Notification sent: {body}."
    return (
        "I couldn't send a notification. "
        "Install libnotify: sudo pacman -S libnotify"
    )


# ══════════════════════════════════════════════════════════════════════════
# ACTIVE WINDOW DETECTION
# ══════════════════════════════════════════════════════════════════════════

def _get_active_window_xdotool() -> str | None:
    """Use xdotool to get active window title (X11 / XWayland)."""
    if not shutil.which("xdotool"):
        return None
    try:
        win_id = subprocess.check_output(
            ["xdotool", "getactivewindow"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode().strip()
        title = subprocess.check_output(
            ["xdotool", "getwindowname", win_id],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return title or None
    except Exception as e:
        log.debug("xdotool window detection failed: %s", e)
    return None


def _get_active_window_kwin() -> str | None:
    """KWin D-Bus scripting to get active window caption (Wayland-native)."""
    qdbus = shutil.which("qdbus6") or shutil.which("qdbus-qt6") or shutil.which("qdbus")
    if not qdbus:
        return None
    try:
        result = subprocess.check_output(
            [qdbus, "org.kde.KWin", "/KWin",
             "org.kde.KWin.activeWindow"],
            timeout=4, stderr=subprocess.DEVNULL,
        ).decode().strip()
        # result is a window ID hex string
        if not result or result == "0":
            return None
        # Get caption via scripting
        script = """
var w = workspace.activeWindow();
print(w ? w.caption : "");
""".strip()
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
            tmp.write(script)
            script_path = tmp.name

        load = subprocess.run(
            [qdbus, "org.kde.KWin", "/Scripting",
             "org.kde.kwin.Scripting.loadScript",
             script_path, f"iris_active_win_{os.getpid()}"],
            timeout=4, check=True, text=True, capture_output=True,
        )
        sid_m = re.search(r"\d+", load.stdout or "")
        if not sid_m:
            return None
        sid = sid_m.group(0)
        output = subprocess.check_output(
            [qdbus, "org.kde.KWin", f"/Scripting/Script{sid}",
             "org.kde.kwin.Script.run"],
            timeout=4, stderr=subprocess.DEVNULL,
        ).decode().strip()
        try:
            os.unlink(script_path)
        except Exception:
            pass
        return output or None
    except Exception as e:
        log.debug("KWin active window detection failed: %s", e)
    return None


def get_active_window() -> str | None:
    """Try all active window detection methods."""
    return _get_active_window_xdotool() or _get_active_window_kwin()


def _match_active_window(norm: str) -> bool:
    return any(p in norm for p in [
        "what am i working on", "what window", "what's focused", "whats focused",
        "active window", "what app is open", "what are you seeing",
        "what's in focus", "whats in focus", "current window",
        "what program", "what application is active",
    ])


def _handle_active_window(command: str, ctx: dict) -> str:
    title = get_active_window()
    if title:
        return f"You're currently in: {title}."
    return (
        "I couldn't detect the active window. "
        "Install xdotool for X11/XWayland: sudo pacman -S xdotool"
    )


# ══════════════════════════════════════════════════════════════════════════
# MONITOR / DISPLAY SWITCHING
# ══════════════════════════════════════════════════════════════════════════

def _list_outputs_kscreen() -> list[str]:
    """List connected outputs via kscreen-doctor."""
    if not shutil.which("kscreen-doctor"):
        return []
    try:
        out = subprocess.check_output(
            ["kscreen-doctor", "-o"], timeout=5, stderr=subprocess.DEVNULL,
        ).decode()
        outputs = []
        for line in out.splitlines():
            # Lines like: "Output: 1 eDP-1 enabled connected"
            m = re.search(r"Output:\s*(\d+)\s+(\S+)", line)
            if m:
                outputs.append(f"{m.group(1)}: {m.group(2)}")
        return outputs
    except Exception:
        return []


def _set_monitor(action: str, target: str) -> bool:
    """Enable/disable/toggle a monitor via kscreen-doctor."""
    if not shutil.which("kscreen-doctor"):
        return False
    try:
        subprocess.run(
            ["kscreen-doctor", f"output.{target}.{action}"],
            check=True, timeout=6,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        log.debug("kscreen-doctor failed: %s", e)
    return False


def _parse_monitor_command(norm: str) -> tuple[str, str] | None:
    """Return (action, target_output) or None."""
    # Patterns: "switch to monitor 2", "turn off display 1", "enable hdmi"
    patterns = [
        (r"\b(enable|turn on|switch to|activate)\s+(?:monitor|display|screen|output)?\s*(\w+)", "enable"),
        (r"\b(disable|turn off|deactivate)\s+(?:monitor|display|screen|output)?\s*(\w+)",       "disable"),
        (r"\b(?:monitor|display|screen)\s+(\w+)\s+(on|off|enable|disable)",                     None),
    ]
    for pat, default_action in patterns:
        m = re.search(pat, norm)
        if m:
            if default_action is None:
                target = m.group(1)
                action = "enable" if m.group(2) in ("on", "enable") else "disable"
            else:
                action = default_action
                target = m.group(2)
            return action, target
    return None


def _match_monitor(norm: str) -> bool:
    return any(p in norm for p in [
        "switch to monitor", "switch monitor", "turn off display",
        "turn on display", "enable monitor", "disable monitor",
        "enable display", "disable display", "turn off monitor",
        "turn on monitor", "list monitors", "list displays",
        "what monitors", "which monitors",
    ])


def _handle_monitor(command: str, ctx: dict) -> str:
    norm = re.sub(r"[^a-z0-9\s]", " ", command.lower())

    if any(w in norm for w in ["list", "what", "which"]):
        outputs = _list_outputs_kscreen()
        if outputs:
            return "Connected displays: " + ", ".join(outputs) + "."
        return (
            "I couldn't list displays. "
            "Install kscreen: sudo pacman -S kscreen"
        )

    parsed = _parse_monitor_command(norm)
    if not parsed:
        return "Which monitor do you want to switch? Say: enable monitor 1, or turn off display 2."

    action, target = parsed
    ok = _set_monitor(action, target)
    if ok:
        return f"Display {target} {action}d."
    return (
        f"I couldn't {action} display {target}. "
        "Make sure kscreen-doctor is installed: sudo pacman -S kscreen"
    )


# ══════════════════════════════════════════════════════════════════════════
# LOCK SCREEN
# ══════════════════════════════════════════════════════════════════════════

def _match_lock(norm: str) -> bool:
    return any(p in norm for p in [
        "lock the screen", "lock screen", "lock my screen",
        "lock the computer", "lock computer",
    ])


def _handle_lock(command: str, ctx: dict) -> str:
    # KDE Plasma
    qdbus = shutil.which("qdbus6") or shutil.which("qdbus-qt6") or shutil.which("qdbus")
    if qdbus:
        try:
            subprocess.run(
                [qdbus, "org.freedesktop.ScreenSaver",
                 "/ScreenSaver", "Lock"],
                check=True, timeout=5,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return "Screen locked."
        except Exception:
            pass
    # loginctl fallback
    if shutil.which("loginctl"):
        try:
            subprocess.run(["loginctl", "lock-session"],
                           check=True, timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "Screen locked."
        except Exception:
            pass
    return "I couldn't lock the screen."


# ══════════════════════════════════════════════════════════════════════════
# SAVE SCREENSHOT
# ══════════════════════════════════════════════════════════════════════════

def _match_save_screenshot(norm: str) -> bool:
    return (
        ("screenshot" in norm or "screen shot" in norm)
        and any(w in norm for w in ["save", "take", "capture", "snap"])
        and "screen" in norm        # avoid matching "take a photo" etc.
    )


def _handle_save_screenshot(command: str, ctx: dict) -> str:
    # Import vision skill's take_screenshot if available
    try:
        from skills.skill_vision import take_screenshot
    except ImportError:
        try:
            import importlib.util, os as _os
            _skills_dir = _os.path.join(_os.path.dirname(__file__))
            spec = importlib.util.spec_from_file_location(
                "skill_vision", _os.path.join(_skills_dir, "skill_vision.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            take_screenshot = mod.take_screenshot
        except Exception:
            take_screenshot = None

    png = take_screenshot() if take_screenshot else None
    if not png:
        return "I couldn't take a screenshot. Try: sudo pacman -S grim"

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"iris-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
    dest = SCREENSHOT_DIR / filename
    dest.write_bytes(png)
    _send_notification("Screenshot saved", str(dest))
    return f"Screenshot saved to {dest}."


# ══════════════════════════════════════════════════════════════════════════
# INTENTS
# ══════════════════════════════════════════════════════════════════════════

INTENTS = [
    {"name": "notify",          "priority": 22, "match": _match_notify,          "handle": _handle_notify},
    {"name": "active_window",   "priority": 22, "match": _match_active_window,   "handle": _handle_active_window},
    {"name": "monitor",         "priority": 22, "match": _match_monitor,         "handle": _handle_monitor},
    {"name": "lock_screen",     "priority": 22, "match": _match_lock,            "handle": _handle_lock},
    {"name": "save_screenshot", "priority": 22, "match": _match_save_screenshot, "handle": _handle_save_screenshot},
]
