"""
skill_system.py — System control intents
Exit, standby, greetings, self-introduction, time, battery, status.
Priority 10 — checked before everything else.
"""

import re
import difflib
import psutil
from datetime import datetime

METADATA = {
    "name":        "System",
    "version":     "1.0",
    "description": "Exit, standby, greetings, time, battery, status, self-intro",
    "author":      "iris",
}

# ── helpers ────────────────────────────────────────────────────────────────

def _contains(text, phrase):
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def _is_exit(norm):
    phrases = {
        "goodbye", "good bye", "goodbye iris", "bye", "bye bye",
        "exit", "exit iris", "quit", "shutdown", "shut down",
        "farewell", "terminate yourself", "shut yourself down",
    }
    return any(norm == p or _contains(norm, p) for p in phrases)


def _is_standby(norm):
    phrases = {
        "sleep", "sleep iris", "iris sleep", "standby", "stand by",
        "go to standby", "go to sleep", "go to standby mode",
    }
    if any(norm == p or _contains(norm, p) for p in phrases):
        return True
    words = norm.split()
    for phrase in phrases:
        pw = phrase.split()
        for n in range(max(1, len(pw) - 1), min(len(words), len(pw) + 1) + 1):
            for i in range(len(words) - n + 1):
                if difflib.SequenceMatcher(None, " ".join(words[i:i+n]), phrase).ratio() >= 0.78:
                    return True
    return False


# ── intents ────────────────────────────────────────────────────────────────

def _match_exit(norm):     return _is_exit(norm)
def _match_standby(norm):  return _is_standby(norm)
def _match_time(norm):     return ("what time" in norm) or ("time" in norm and "what" in norm)
def _match_status(norm):   return any(w in norm for w in ["battery", "cpu", "status"])
def _match_intro(norm):    return any(p in norm for p in ["introduce yourself", "who are you", "tell me about yourself"])
def _match_location(norm): return any(p in norm for p in ["where do i live", "where i live", "my location", "where am i from"])
def _match_memory_summary(norm): return any(p in norm for p in ["what do you know", "what do you remember", "what you know about me"])
def _match_forget_all(norm): return any(p in norm for p in ["forget everything", "clear memory", "reset memory"])
def _match_forget_last(norm): return "forget that" in norm
def _match_reload_skills(norm): return norm in ("reload skills", "refresh skills", "load skills")


def _handle_exit(command, ctx):
    ctx["shutdown"]()
    return None


def _handle_standby(command, ctx):
    ctx["set_active"](False)
    name = ctx["memory"]["user"].get("name") or ""
    suffix = f", {name}" if name else ""
    return f"Going to standby. Say 'Hey Iris' when you need me{suffix}."


def _handle_time(command, ctx):
    return f"It is {datetime.now().strftime('%I:%M %p')}."


def _handle_status(command, ctx):
    battery = psutil.sensors_battery()
    battery_str = (
        f"{battery.percent:.0f}% {'charging' if battery.power_plugged else 'on battery'}"
        if battery else "unknown"
    )
    return (
        f"Time is {datetime.now().strftime('%I:%M %p')}. "
        f"Battery at {battery_str}. "
        f"CPU at {psutil.cpu_percent()}%."
    )


def _handle_intro(command, ctx):
    name = ctx["user_name"]
    return (
        f"I am Iris, your personal assistant, {name}. "
        "I can control music, system actions, web tasks, and remember important details about you."
    )


def _handle_location(command, ctx):
    loc = ctx["memory"].get("user", {}).get("location")
    if loc:
        return f"You live in {loc}."
    return "I do not have your location saved yet. Tell me: I live in your city."


def _handle_memory_summary(command, ctx):
    import iris_memory as mem
    return mem.get_memory_summary(ctx["memory"])


def _handle_forget_all(command, ctx):
    import iris_memory as mem
    mem.clear_memory(ctx["memory"])
    # reload into the live memory dict in-place
    fresh = mem.load()
    ctx["memory"].clear()
    ctx["memory"].update(fresh)
    return "Memory wiped. Starting fresh."


def _handle_forget_last(command, ctx):
    import iris_memory as mem
    if ctx["memory"]["facts"]:
        removed = ctx["memory"]["facts"].pop()
        mem.save(ctx["memory"])
        return f"Forgotten: {removed}."
    return "Nothing recent to forget."


def _handle_reload_skills(command, ctx):
    import iris_skills
    count = iris_skills.reload_skills()
    return f"Skills reloaded. {count} intents active."


INTENTS = [
    {"name": "exit",           "priority": 5,  "match": _match_exit,           "handle": _handle_exit},
    {"name": "standby",        "priority": 5,  "match": _match_standby,        "handle": _handle_standby},
    {"name": "time",           "priority": 10, "match": _match_time,           "handle": _handle_time},
    {"name": "status",         "priority": 10, "match": _match_status,         "handle": _handle_status},
    {"name": "intro",          "priority": 15, "match": _match_intro,          "handle": _handle_intro},
    {"name": "location",       "priority": 15, "match": _match_location,       "handle": _handle_location},
    {"name": "memory_summary", "priority": 15, "match": _match_memory_summary, "handle": _handle_memory_summary},
    {"name": "forget_all",     "priority": 15, "match": _match_forget_all,     "handle": _handle_forget_all},
    {"name": "forget_last",    "priority": 15, "match": _match_forget_last,    "handle": _handle_forget_last},
    {"name": "reload_skills",  "priority": 20, "match": _match_reload_skills,  "handle": _handle_reload_skills},
]
