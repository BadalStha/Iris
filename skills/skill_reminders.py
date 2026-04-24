"""
skill_reminders.py — Reminder parsing and saving
Priority 30.
"""

import re
from datetime import datetime

import iris_memory as mem

METADATA = {
    "name":        "Reminders",
    "version":     "1.0",
    "description": "Set time-based reminders by voice",
    "author":      "iris",
}


def _parse(command):
    now = datetime.now()
    cmd = command.lower()

    m = re.search(
        r"remind me in (\d+)\s*(minute|minutes|min|hour|hours|hr)\s*(?:to\s+)?(.+)",
        cmd,
    )
    if m:
        amt, unit, text = int(m.group(1)), m.group(2), m.group(3).strip().rstrip(".,?!")
        delta = amt * (3600 if "hour" in unit or "hr" in unit else 60)
        at = datetime.fromtimestamp(now.timestamp() + delta)
        return text, at.strftime("%Y-%m-%d %H:%M")

    m = re.search(
        r"remind me at (\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to\s+)?(.+)",
        cmd,
    )
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        meridiem, text = m.group(3), m.group(4).strip().rstrip(".,?!")
        if meridiem == "pm" and hour != 12: hour += 12
        elif meridiem == "am" and hour == 12: hour = 0
        at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if at < now:
            at = at.replace(day=at.day + 1)
        return text, at.strftime("%Y-%m-%d %H:%M")

    return None, None


def _match_reminder(norm):
    return "remind me" in norm


def _handle_reminder(command, ctx):
    text, at = _parse(command)
    if text and at:
        mem.save_reminder(ctx["memory"], text, at)
        dt = datetime.strptime(at, "%Y-%m-%d %H:%M")
        return f"Got it. I'll remind you to {text} at {dt.strftime('%I:%M %p')}."
    return "When should I remind you, and what about?"


INTENTS = [
    {"name": "set_reminder", "priority": 30, "match": _match_reminder, "handle": _handle_reminder},
]
