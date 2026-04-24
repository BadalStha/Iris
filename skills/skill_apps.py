"""
skill_apps.py — Application launcher and closer
Handles: open X, launch X, close X, quit X.
Priority 25.
"""

import difflib
import re
import shutil
import subprocess
from pathlib import Path

import psutil

METADATA = {
    "name":        "Apps",
    "version":     "1.0",
    "description": "Open and close desktop applications",
    "author":      "iris",
}

APP_ALIASES = {
    "vs code": {
        "command": ["code"],
        "aliases": ["vscode", "visual studio code", "bs code", "bc code", "v code", "visual code"],
    },
    "firefox": {
        "command": ["firefox"],
        "aliases": ["firebox", "fire fox", "firefox browser"],
    },
    "discord": {
        "command": ["discord"],
        "aliases": ["discord app"],
    },
    "steam": {
        "command": ["steam"],
        "aliases": ["steam app"],
    },
    "intellij": {
        "command": ["idea", "intellij-idea-community-edition", "intellij-idea-ultimate-edition"],
        "aliases": ["intelij", "inteli j", "intellij idea", "idea", "jetbrains idea"],
    },
    "terminal": {
        "command": ["konsole", "gnome-terminal", "xterm"],
        "aliases": ["bash", "shell", "console", "command line", "term"],
    },
    "file manager": {
        "command": ["dolphin", "nautilus", "thunar"],
        "aliases": ["file explorer", "files", "dolphin", "the door thing", "door thing"],
    },
}

_OPEN_WORDS  = ["open", "launch", "start", "run"]
_CLOSE_WORDS = ["close", "quit", "terminate", "end"]

_STRIP_OPEN  = re.compile(
    r"\b(open|launch|start|can you|could you|please|me|i said|just|now|for me|run|the|an|a|app|application)\b",
    re.IGNORECASE,
)
_STRIP_CLOSE = re.compile(
    r"\b(close|quit|terminate|end|please|can you|could you|for me|the|app|application|i said|just|now)\b",
    re.IGNORECASE,
)


# ── lookup ─────────────────────────────────────────────────────────────────

def _norm(text):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split())


def _find_app(raw_name, memory=None):
    normalized = _norm(raw_name)
    if not normalized:
        return None, None

    if memory:
        for learned, display in memory.get("app_corrections", {}).items():
            if normalized == learned or difflib.SequenceMatcher(None, normalized, learned).ratio() >= 0.80:
                res = _lookup(display.lower())
                if res[0]:
                    return res

    return _lookup(normalized)


def _lookup(normalized):
    for display, info in APP_ALIASES.items():
        if normalized in (display, display.replace(" ", "")):
            return display, info["command"]
    for display, info in APP_ALIASES.items():
        for alias in info["aliases"]:
            a = _norm(alias)
            if normalized in (a, a.replace(" ", "")):
                return display, info["command"]
            if difflib.SequenceMatcher(None, normalized, a).ratio() >= 0.75:
                return display, info["command"]
    for display, info in APP_ALIASES.items():
        if difflib.SequenceMatcher(None, normalized, display).ratio() >= 0.70:
            return display, info["command"]
    return None, None


def _resolve_cmd(command_list):
    for cmd in (command_list if isinstance(command_list, list) else [command_list]):
        if cmd and shutil.which(cmd):
            return cmd
    return None


# ── match ──────────────────────────────────────────────────────────────────

def _match_open(norm):
    if any(w in norm for w in _OPEN_WORDS):
        return True
    display, _ = _find_app(norm)
    return bool(display) and len(norm.split()) <= 5


def _match_close(norm):
    if not any(w in norm for w in _CLOSE_WORDS):
        return False
    cleaned = _norm(_STRIP_CLOSE.sub(" ", norm))
    display, _ = _find_app(cleaned)
    return bool(display)


# ── handle ─────────────────────────────────────────────────────────────────

def _handle_open(command, ctx):
    cleaned = _norm(_STRIP_OPEN.sub(" ", command))
    if not cleaned:
        return "Which app would you like to open?"

    memory = ctx["memory"]
    display, cmd_list = _find_app(cleaned, memory)
    if not display:
        return (
            f"I couldn't find an app called {cleaned}. "
            "Try: Discord, File Manager, Steam, Firefox, Terminal, VS Code, or IntelliJ."
        )
    cmd = _resolve_cmd(cmd_list)
    if not cmd:
        return f"{display} doesn't seem to be installed."

    reply = f"Opening {display}." if cleaned.lower() == display.lower() else f"I think you meant {display}. Opening it now."
    try:
        subprocess.Popen(
            [cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=__import__("os").setsid if hasattr(__import__("os"), "setsid") else None,
        )
    except FileNotFoundError:
        return f"{display} is not installed."
    except Exception as exc:
        return f"Failed to open {display}: {str(exc)[:60]}"
    return reply


def _handle_close(command, ctx):
    cleaned = _norm(_STRIP_CLOSE.sub(" ", command))
    if not cleaned:
        return "Which app should I close?"

    memory = ctx["memory"]
    display, cmd_list = _find_app(cleaned, memory)
    if not display:
        return f"I couldn't find an app called {cleaned} to close."
    if display == "terminal":
        return "I won't close terminal automatically."

    cmds = cmd_list if isinstance(cmd_list, list) else [cmd_list]
    hints = {display.lower().replace(" ", "")}
    for c in cmds:
        if c:
            hints.add(c.lower())
            hints.add(Path(c).name.lower())

    matched = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if any(h and (h in name or h in cmdline) for h in hints):
                matched.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not matched:
        return f"{display} is not running right now."

    closed = 0
    for proc in matched:
        try:
            proc.terminate()
            closed += 1
        except Exception:
            pass
    return f"Closed {display}." if closed else f"I couldn't close {display} due to permissions."


INTENTS = [
    {"name": "open_app",  "priority": 25, "match": _match_open,  "handle": _handle_open},
    {"name": "close_app", "priority": 25, "match": _match_close, "handle": _handle_close},
]
