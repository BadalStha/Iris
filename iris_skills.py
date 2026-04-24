"""
iris_skills.py — Iris Plugin / Skill Loader
============================================
Scans the ./skills/ folder for skill_*.py files, imports each one,
and exposes a single dispatch function used by iris.py.

HOW A SKILL FILE WORKS
-----------------------
Each skill_*.py must define two things:

    INTENTS: list[dict]
        A list of intent definitions.  Each dict has:
          - "name"     : str   — unique label e.g. "play_music"
          - "match"    : callable(normalized_command: str) -> bool
                         Return True when this skill should handle the command.
          - "handle"   : callable(command: str, context: dict) -> str | None
                         Do the work.  Return a spoken reply string, or None
                         if the skill speaks for itself (e.g. calls speak() directly).
          - "priority" : int   (optional, default 50)
                         Lower number = checked first.  Built-in fallback = 999.

    METADATA: dict
        {
          "name":        "Human-readable skill name",
          "version":     "1.0",
          "description": "One line about what this skill does",
          "author":      "you",
        }

CONTEXT DICT
------------
The context dict passed to every handle() call contains:
    context["memory"]       — the live memory dict (read/write)
    context["speak"]        — the speak(text) function
    context["user_name"]    — current user name string
    context["is_active"]    — bool, whether Iris is in active mode
    context["set_active"]   — callable(bool) to change active state
    context["shutdown"]     — callable() to exit Iris

Skills can import and use anything from iris_memory, stdlib, etc.
They should NOT import from iris.py (circular).

EXAMPLE SKILL
-------------
See skills/skill_music.py for a complete example.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# Each entry: {"name", "match", "handle", "priority", "_source"}
_registered_intents: list[dict] = []
_loaded_skill_names: list[str]  = []


# ── public API ─────────────────────────────────────────────────────────────

def load_skills() -> None:
    """
    Import every skills/skill_*.py file.
    Safe to call multiple times — reloads changed files.
    """
    global _registered_intents, _loaded_skill_names
    _registered_intents = []
    _loaded_skill_names = []

    if not _SKILLS_DIR.exists():
        log.warning("Skills directory not found: %s", _SKILLS_DIR)
        return

    skill_files = sorted(_SKILLS_DIR.glob("skill_*.py"))
    if not skill_files:
        log.info("No skill files found in %s", _SKILLS_DIR)
        return

    for skill_path in skill_files:
        _load_one(skill_path)

    # Sort by priority so lower-number skills are checked first
    _registered_intents.sort(key=lambda x: x.get("priority", 50))

    log.info(
        "Skills loaded: %s  (%d intents)",
        ", ".join(_loaded_skill_names),
        len(_registered_intents),
    )


def dispatch(command: str, context: dict) -> tuple[bool, Optional[str]]:
    """
    Try every registered skill intent in priority order.

    Returns:
        (handled: bool, reply: str | None)
        handled=True  means a skill matched and ran.
        reply is the string to speak, or None if skill spoke itself.
    """
    normalized = _normalize(command)

    for intent in _registered_intents:
        try:
            if intent["match"](normalized):
                log.info("Skill dispatch: %s → %s", intent["_source"], intent["name"])
                result = intent["handle"](command, context)
                return True, result
        except Exception as exc:
            log.warning(
                "Skill '%s' handler raised: %s", intent.get("name", "?"), exc
            )

    return False, None


def list_skills() -> list[dict]:
    """Return metadata for all loaded skills (for debug / 'what can you do')."""
    return [
        {
            "source":   i["_source"],
            "intent":   i["name"],
            "priority": i.get("priority", 50),
        }
        for i in _registered_intents
    ]


def reload_skills() -> int:
    """Hot-reload all skills without restarting Iris.  Returns new intent count."""
    load_skills()
    return len(_registered_intents)


# ── helpers ────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return " ".join(normalized.split())


def _load_one(path: Path) -> None:
    """Import a single skill file and register its intents."""
    module_name = f"iris_skill_{path.stem}"
    try:
        spec   = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        intents  = getattr(module, "INTENTS",  None)
        metadata = getattr(module, "METADATA", {})

        if not intents or not isinstance(intents, list):
            log.warning("Skill %s has no INTENTS list — skipped.", path.name)
            return

        skill_label = metadata.get("name", path.stem)
        _loaded_skill_names.append(skill_label)

        valid = 0
        for intent in intents:
            if not callable(intent.get("match")) or not callable(intent.get("handle")):
                log.warning(
                    "Skill %s intent '%s' missing match/handle — skipped.",
                    path.name, intent.get("name", "?"),
                )
                continue
            intent["_source"] = path.stem
            _registered_intents.append(intent)
            valid += 1

        log.info("Loaded skill '%s' from %s  (%d intents)", skill_label, path.name, valid)

    except Exception as exc:
        log.error("Failed to load skill %s: %s", path.name, exc, exc_info=True)
