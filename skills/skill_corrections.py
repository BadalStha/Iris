"""
skill_corrections.py — Handle user corrections to Iris replies
Priority 40.
"""

import re
import iris_memory as mem

METADATA = {
    "name":        "Corrections",
    "version":     "1.0",
    "description": "Handle 'that was wrong', 'I meant X', 'it's actually Y'",
    "author":      "iris",
}

_CORRECTION_TRIGGERS = [
    "you said", "you meant", "its actually", "it s actually", "it's actually",
    "i said", "i meant", "wrong", "mistake", "correct",
]

_CORRECTION_RE = re.compile(
    r"\bit['\s]*s\s+(.+?)(?:\s+not\s+.+)?$"
    r"|\bnot\s+\S+\s+(?:its?|it['\s]*s)\s+(.+)$"
    r"|(?:i said|i meant)\s+(.+)$"
)


def _match_correction(norm):
    return any(p in norm for p in _CORRECTION_TRIGGERS)


def _handle_correction(command, ctx):
    norm = re.sub(r"[^a-z0-9\s]", " ", command.lower())
    norm = " ".join(norm.split())
    m = _CORRECTION_RE.search(norm)
    if m:
        corrected = next((g for g in m.groups() if g), "").strip()
        if corrected:
            mem.save_correction(ctx["memory"], corrected)
            return f"Got it. I'll remember that you meant {corrected}."
    return "Sorry about that. Could you repeat what you meant?"


INTENTS = [
    {"name": "correction", "priority": 40, "match": _match_correction, "handle": _handle_correction},
]
