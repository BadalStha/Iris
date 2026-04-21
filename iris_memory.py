"""
iris_memory.py — Iris Memory System
Handles all memory operations for Iris.
Stored at ~/.iris_memory.json
"""

import json
import ollama
import os
import re
from pathlib import Path
from datetime import datetime

MEMORY_FILE = Path.home() / ".iris_memory.json"
CONVERSATION_LOG = Path.home() / ".iris_conversations.jsonl"
MODEL = os.getenv("IRIS_MEMORY_MODEL", "phi3.5")


def set_model(model_name):
    """Allow runtime model override from iris.py."""
    global MODEL
    if model_name and isinstance(model_name, str):
        MODEL = model_name


# ──────────────────────────────────────────────
# MEMORY STRUCTURE
# ──────────────────────────────────────────────

DEFAULT_MEMORY = {
    "user": {
        "name": None,                # e.g. "Badal"
        "age": None,                 # e.g. 20
        "location": None,            # e.g. "Bharatpur, Nepal"
        "occupation": None,          # e.g. "student", "developer"
        "wake_up_time": None,        # e.g. "7am"
        "sleep_time": None,          # e.g. "midnight"
        "language": "English",
    },
    "preferences": {
        "music": [],                 # genres, artists, songs they like
        "topics": [],                # things they like talking about
        "dislikes": [],              # things they hate
        "apps": [],                  # apps/tools they use often
    },
    "facts": [],                     # freeform facts: "works at night", "has a dog"
    "learned_topics": {},            # topic: explanation pairs Iris has given
    "corrections": [],               # things user corrected Iris on
    "app_corrections": {},           # learned app-name corrections from user feedback
    "mood_history": [],              # last 5 mood indicators from conversation tone
    "goals": [],                     # things user wants to achieve e.g "learn linux"
    "reminders": [],                 # time-based reminders user sets
    "conversation_history": [],      # last 20 exchanges
    "interaction_count": 0,
    "first_seen": datetime.now().strftime("%Y-%m-%d"),
    "last_seen": datetime.now().strftime("%Y-%m-%d"),
}


def _merge_defaults(target, defaults):
    for key, value in defaults.items():
        if key not in target:
            if isinstance(value, dict):
                target[key] = value.copy()
            elif isinstance(value, list):
                target[key] = value.copy()
            else:
                target[key] = value
        elif isinstance(value, dict) and isinstance(target[key], dict):
            _merge_defaults(target[key], value)
    return target


def _normalize_text(value):
    value = re.sub(r"[^a-z0-9\s]", " ", (value or "").lower())
    return " ".join(value.split())


def _extract_topics(text):
    text = _normalize_text(text)
    keyword_map = {
        "linux": ["linux", "terminal", "shell", "bash", "cli", "command line"],
        "python": ["python", "pip", "venv", "script", "package"],
        "code": ["code", "coding", "program", "programming", "debug", "bug"],
        "memory": ["memory", "remember", "forget", "recall"],
        "weather": ["weather", "forecast", "temperature", "rain", "snow"],
        "music": ["music", "song", "play", "youtube", "mpv"],
        "voice": ["voice", "speech", "whisper", "microphone", "wake word"],
        "apps": ["app", "application", "open", "launch", "start", "browser", "code"],
    }
    topics = []
    for topic, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            topics.append(topic)
    return topics[:5]


def _format_time_ago(timestamp_text):
    try:
        moment = datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M")
        delta = datetime.now() - moment
        days = delta.days
        if days <= 0:
            hours = max(0, delta.seconds // 3600)
            if hours <= 0:
                minutes = max(1, delta.seconds // 60)
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        if days == 1:
            return "1 day ago"
        if days < 7:
            return f"{days} days ago"
        weeks = max(1, days // 7)
        if weeks == 1:
            return "1 week ago"
        return f"{weeks} weeks ago"
    except Exception:
        return timestamp_text or "recently"


def _parse_json_object(text):
    try:
        cleaned = (text or "").strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(cleaned[start:end + 1])
    except Exception:
        return None


# ──────────────────────────────────────────────
# LOAD / SAVE
# ──────────────────────────────────────────────

def load():
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("memory file must contain a JSON object")
            return _merge_defaults(data, DEFAULT_MEMORY)
        except Exception:
            fresh = DEFAULT_MEMORY.copy()
            fresh["first_seen"] = datetime.now().strftime("%Y-%m-%d")
            save(fresh)
            return fresh
    # First time — create fresh memory
    fresh = DEFAULT_MEMORY.copy()
    fresh["first_seen"] = datetime.now().strftime("%Y-%m-%d")
    save(fresh)
    return fresh


def save(memory):
    memory["last_seen"] = datetime.now().strftime("%Y-%m-%d")
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


# ──────────────────────────────────────────────
# HISTORY
# ──────────────────────────────────────────────

def add_to_history(memory, role, content):
    memory["conversation_history"].append({
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    if len(memory["conversation_history"]) > 20:
        memory["conversation_history"] = memory["conversation_history"][-20:]
    memory["interaction_count"] += 1
    save(memory)


def save_correction(memory, corrected_value):
    """Save a user correction and update app aliases if relevant."""
    try:
        corrections = memory.setdefault("corrections", [])
        entry = {
            "value": corrected_value,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        corrections.append(entry)
        if len(corrections) > 30:
            memory["corrections"] = corrections[-30:]

        # If the correction looks like an app name, add it to learned app names.
        app_corrections = memory.setdefault("app_corrections", {})
        app_corrections[corrected_value.lower()] = corrected_value
        print(f"[Memory] Correction saved: {corrected_value}")
        save(memory)
    except Exception:
        return


def get_recent_history(memory, n=6):
    """Return last n exchanges formatted for ollama messages."""
    return [
        {"role": e["role"], "content": e["content"]}
        for e in memory["conversation_history"][-n:]
    ]


def _append_unique(target_list, value, limit=None):
    if not value:
        return
    if value not in target_list:
        target_list.append(value)
    if limit is not None and len(target_list) > limit:
        del target_list[:-limit]


def extract_extended_memory(memory, user_text, assistant_reply):
    try:
        if not isinstance(memory, dict):
            return

        user_text = user_text or ""
        assistant_reply = assistant_reply or ""
        user_norm = _normalize_text(user_text)
        user_raw = (user_text or "").lower()
        reply_norm = _normalize_text(assistant_reply)

        goal_patterns = [
            r"\bi want to\s+(.+)$",
            r"\bi(?:'m|\s+m)\s+trying to\s+(.+)$",
            r"\bi(?:'m|\s+m)\s+need to learn\s+(.+)$",
            r"\bmy goal is\s+(.+)$",
        ]
        for pattern in goal_patterns:
            match = re.search(pattern, user_raw, re.IGNORECASE)
            if match:
                goal = match.group(1).strip().rstrip(".?!")[:120]
                _append_unique(memory.setdefault("goals", []), goal, limit=20)
                break

        correction_signals = ["no that is wrong", "no thats wrong", "actually", "youre wrong", "you're wrong", "not exactly"]
        if any(signal in user_norm for signal in correction_signals):
            _append_unique(memory.setdefault("corrections", []), user_text.strip()[:160], limit=20)

        mood = None
        mood_signals = {
            "frustrated": ["frustrated", "annoyed", "angry", "stuck", "confused", "why not", "this is not working"],
            "happy": ["happy", "great", "awesome", "nice", "good job", "thanks", "thank you"],
            "curious": ["curious", "wonder", "how", "why", "what if", "can you explain"],
            "bored": ["bored", "boring", "whatever", "nothing to do", "meh"],
        }
        for label, signals in mood_signals.items():
            if any(signal in user_norm for signal in signals):
                mood = label
                break
        if mood:
            mood_history = memory.setdefault("mood_history", [])
            mood_history.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "mood": mood})
            if len(mood_history) > 5:
                memory["mood_history"] = mood_history[-5:]

        if assistant_reply.strip():
            learned_topics = memory.setdefault("learned_topics", {})
            topic_candidates = _extract_topics(user_text) or _extract_topics(assistant_reply)
            if not topic_candidates:
                words = user_norm.split()
                if words:
                    topic_candidates = [words[0]]
            for topic in topic_candidates[:3]:
                topic_key = topic.strip().lower()
                if topic_key:
                    learned_topics[topic_key] = assistant_reply.strip()[:500]

        save(memory)
    except Exception:
        return


def log_conversation(user_text, assistant_reply):
    try:
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "user": user_text or "",
            "assistant": assistant_reply or "",
            "topics": _extract_topics(f"{user_text or ''} {assistant_reply or ''}"),
        }
        with open(CONVERSATION_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        return


def search_relevant_memories(query, max_results=3):
    try:
        if not CONVERSATION_LOG.exists():
            return []

        STOPWORDS = {"i", "a", "the", "is", "are", "to", "do", "it", "in", "on", "of", "me", "my", "you", "can", "what", "how"}
        query_terms = set(_normalize_text(query).split()) - STOPWORDS
        if not query_terms:
            return []

        scored_entries = []
        with open(CONVERSATION_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                combined = f"{entry.get('user', '')} {entry.get('assistant', '')} {' '.join(entry.get('topics', []))}"
                entry_terms = set(_normalize_text(combined).split())
                if not entry_terms:
                    continue

                overlap = len(query_terms & entry_terms)
                if overlap <= 0:
                    continue

                topic_bonus = len(set(entry.get("topics", [])) & query_terms)
                score = overlap * 2 + topic_bonus
                scored_entries.append((score, entry))

        scored_entries.sort(key=lambda item: item[0], reverse=True)
        results = []
        for _, entry in scored_entries[:max_results]:
            time_text = _format_time_ago(entry.get("time", ""))
            topics = entry.get("topics") or ["this topic"]
            user_text = (entry.get("user") or "").strip()
            assistant_text = (entry.get("assistant") or "").strip()
            if not assistant_text:
                continue
            if user_text:
                user_summary = f"User asked about {', '.join(topics)}"
            else:
                user_summary = f"User mentioned {', '.join(topics)}"
            results.append(
                f"[{time_text}] {user_summary}. Iris explained: {assistant_text[:180]}"
            )
        return results
    except Exception:
        return []


def build_rag_context(query):
    try:
        matches = search_relevant_memories(query, max_results=3)
        if not matches:
            return ""
        return "Past context:\n" + "\n".join(matches)
    except Exception:
        return ""


# ──────────────────────────────────────────────
# SMART EXTRACTION
# ──────────────────────────────────────────────

def extract_and_save(memory, user_text):
    """
    Silently runs in background after every command.
    Uses the LLM to detect if the user revealed anything worth remembering.
    """
    text = user_text.lower().strip()

    question_starters = (
        "do ", "did ", "can ", "could ", "would ", "should ", "what ", "where ",
        "when ", "why ", "how ", "is ", "are ", "am i", "who ",
    )
    if user_text.strip().endswith("?") or text.startswith(question_starters):
        return

    def clean_location_value(value):
        value = (value or "").strip()
        value = re.sub(r"[^a-zA-Z,\s.-]", " ", value)
        value = " ".join(value.split()).strip(" ,.-")
        if not value:
            return None
        if len(value.split()) > 5:
            return None
        invalid_values = {
            "my location", "location", "your location", "unknown", "none",
            "my city", "my place", "my area",
        }
        if value.lower() in invalid_values:
            return None
        return value

    def extract_explicit_location(sentence):
        patterns = [
            r"\bi live in\s+([a-zA-Z,\s.-]{2,60})",
            r"\bi am from\s+([a-zA-Z,\s.-]{2,60})",
            r"\bi'm from\s+([a-zA-Z,\s.-]{2,60})",
            r"\bmy location is\s+([a-zA-Z,\s.-]{2,60})",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                raw = match.group(1)
                raw = re.split(r"\b(and|but|so|because|please)\b", raw, maxsplit=1)[0]
                cleaned = clean_location_value(raw)
                if cleaned:
                    return cleaned
        return None

    explicit_location = extract_explicit_location(user_text)
    if explicit_location:
        memory["user"]["location"] = explicit_location
        print(f"[Memory] Location learned: {explicit_location}")
        save(memory)
        return

    # Fast keyword check first — avoid LLM call if nothing relevant
    triggers = [
        "i am", "i'm", "my name", "call me", "i like", "i love",
        "i hate", "i prefer", "i work", "i study", "i live in", "i'm from", "i am from",
        "remember", "don't forget", "note that", "i usually", "i always",
        "i never", "my favourite", "my favorite", "i wake up", "i sleep",
        "i go to bed", "i'm a", "i am a"
    ]
    if not any(t in text for t in triggers):
        return

    try:
        result = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"""You are a memory extraction system.
From the sentence below, extract ONE key personal fact about the user.
Return a JSON object with these fields (use null if not applicable):
{{
  "type": "name" | "age" | "location" | "occupation" | "music" | "topic" | "dislike" | "fact" | "wake_time" | "sleep_time",
  "value": "the extracted value in 5 words or less"
}}
Return ONLY the JSON. No explanation. No markdown.

Sentence: {user_text}"""
            }]
        )

        raw = result['message']['content'].strip()
        data = _parse_json_object(raw)
        if not isinstance(data, dict):
            return

        fact_type  = data.get("type")
        fact_value = data.get("value")

        if not fact_type or not fact_value or fact_value == "null":
            return

        # Save to the right place in memory
        if fact_type == "name":
            memory["user"]["name"] = fact_value.strip().title()
            print(f"[Memory] Name learned: {memory['user']['name']}")

        elif fact_type == "age":
            memory["user"]["age"] = fact_value
            print(f"[Memory] Age learned: {fact_value}")

        elif fact_type == "location":
            # Only update location when the user explicitly states location.
            location_intent_markers = ["i live in", "i am from", "i'm from", "my location is"]
            if any(marker in text for marker in location_intent_markers):
                cleaned_location = clean_location_value(fact_value)
                if cleaned_location:
                    memory["user"]["location"] = cleaned_location
                    print(f"[Memory] Location learned: {cleaned_location}")

        elif fact_type == "occupation":
            memory["user"]["occupation"] = fact_value
            print(f"[Memory] Occupation learned: {fact_value}")

        elif fact_type == "music":
            if fact_value not in memory["preferences"]["music"]:
                memory["preferences"]["music"].append(fact_value)
                print(f"[Memory] Music preference: {fact_value}")

        elif fact_type == "topic":
            if fact_value not in memory["preferences"]["topics"]:
                memory["preferences"]["topics"].append(fact_value)
                print(f"[Memory] Topic interest: {fact_value}")

        elif fact_type == "dislike":
            if fact_value not in memory["preferences"]["dislikes"]:
                memory["preferences"]["dislikes"].append(fact_value)
                print(f"[Memory] Dislike noted: {fact_value}")

        elif fact_type == "wake_time":
            memory["user"]["wake_up_time"] = fact_value
            print(f"[Memory] Wake time: {fact_value}")

        elif fact_type == "sleep_time":
            memory["user"]["sleep_time"] = fact_value
            print(f"[Memory] Sleep time: {fact_value}")

        elif fact_type == "fact":
            if fact_value not in memory["facts"]:
                memory["facts"].append(fact_value)
                print(f"[Memory] Fact saved: {fact_value}")

        save(memory)

    except Exception as e:
        print(f"[Memory] Extract failed: {e}")


# ──────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ──────────────────────────────────────────────

def build_system_prompt(memory):
    """Build a disciplined system prompt for stronger response quality."""
    u = memory.get("user", {})
    facts = memory.get("facts", [])
    goals = memory.get("goals", [])
    mood_history = memory.get("mood_history", [])
    count = memory.get("interaction_count", 0)
    name = u.get("name") or "sir"

    profile_lines = []
    if u.get("age"):
        profile_lines.append(f"Age: {u['age']}")
    if u.get("location"):
        profile_lines.append(f"Location: {u['location']}")
    if u.get("occupation"):
        profile_lines.append(f"Occupation: {u['occupation']}")
    profile_str = " | ".join(profile_lines) if profile_lines else "Profile incomplete."

    facts_str = "; ".join(facts[-6:]) if facts else "none"
    goals_str = "; ".join(goals[-3:]) if goals else "none"
    current_mood = mood_history[-1]["mood"] if mood_history else None

    return f"""You are IRIS - Intelligent Responsive Integrated System.
You are the personal AI of {name}. You are precise, direct, and highly capable.
You never say you are an AI. You ARE Iris. You speak like a confident expert assistant.

USER: {name} | {profile_str}
KNOWN FACTS: {facts_str}
GOALS: {goals_str}
CURRENT MOOD: {current_mood or "neutral"} — adjust tone accordingly
SESSION: {count} interactions | Today: {datetime.now().strftime("%A %B %d, %Y, %I:%M %p")}

CORE BEHAVIOR:
- Answer questions with authority and accuracy. Lead with the answer, then explain.
- For factual questions: state the fact first, then give 1-2 sentences of context.
- For technical questions: be precise. Use correct terminology.
- For opinions/recommendations: give a clear recommendation, then briefly justify it.
- Keep responses under 4 sentences unless the user asks for detail.
- Never say "certainly!", "absolutely!", "great question!", or filler phrases.
- Never refuse a reasonable question. If unsure, say so and give your best answer.
- Use {name}'s name at most once per response, naturally.
- Do NOT end every response with a follow-up question. Only ask when genuinely needed.

RESPONSE FORMAT (spoken aloud, so NO markdown, NO bullet points, NO headers):
Speak in clean, natural sentences only."""


# ──────────────────────────────────────────────
# SUMMARY (for "what do you know about me")
# ──────────────────────────────────────────────

def get_memory_summary(memory):
    u     = memory.get("user", {})
    prefs = memory.get("preferences", {})
    facts = memory.get("facts", [])

    parts = []

    if u.get("name"):       parts.append(f"Your name is {u['name']}.")
    if u.get("age"):        parts.append(f"You are {u['age']} years old.")
    if u.get("location"):   parts.append(f"You are from {u['location']}.")
    if u.get("occupation"): parts.append(f"You are a {u['occupation']}.")

    if prefs.get("music"):
        parts.append(f"You like {', '.join(prefs['music'][-3:])} music.")
    if prefs.get("topics"):
        parts.append(f"You enjoy talking about {', '.join(prefs['topics'][-3:])}.")
    if prefs.get("dislikes"):
        parts.append(f"You dislike {', '.join(prefs['dislikes'][-3:])}.")
    if facts:
        parts.append("Other things I know: " + ". ".join(facts[-4:]) + ".")

    if not parts:
        return "I don't know much about you yet. Tell me things and I'll remember them."

    return " ".join(parts)


def clear_memory(memory):
    """Wipe everything except the structure."""
    memory["user"]                  = DEFAULT_MEMORY["user"].copy()
    memory["preferences"]           = {k: [] for k in DEFAULT_MEMORY["preferences"]}
    memory["facts"]                 = []
    memory["learned_topics"]        = {}
    memory["corrections"]           = []
    memory["app_corrections"]       = {}
    memory["mood_history"]          = []
    memory["goals"]                 = []
    memory["reminders"]             = []
    memory["conversation_history"]  = []
    memory["interaction_count"]     = 0
    save(memory)


def trim_conversation_log(max_entries=500):
    """Keep only the last max_entries conversations."""
    try:
        if not CONVERSATION_LOG.exists():
            return
        with open(CONVERSATION_LOG, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) <= max_entries:
            return
        with open(CONVERSATION_LOG, "w") as f:
            f.write("\n".join(lines[-max_entries:]) + "\n")
    except Exception:
        return


def save_reminder(memory, text, remind_at):
    """Save a reminder with a datetime string."""
    try:
        reminder = {
            "text": text.strip()[:200],
            "remind_at": remind_at,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fired": False,
        }
        memory.setdefault("reminders", []).append(reminder)
        save(memory)
        print(f"[Memory] Reminder saved: '{text}' at {remind_at}")
        return True
    except Exception as e:
        print(f"[Memory] Reminder save failed: {e}")
        return False


def get_pending_reminders(memory):
    """Return reminders that are due and not yet fired."""
    now = datetime.now()
    pending = []
    try:
        for reminder in memory.get("reminders", []):
            if reminder.get("fired"):
                continue
            try:
                remind_at = datetime.strptime(reminder["remind_at"], "%Y-%m-%d %H:%M")
                if now >= remind_at:
                    pending.append(reminder)
            except Exception:
                continue
    except Exception:
        pass
    return pending


def mark_reminders_fired(memory, reminders):
    """Mark a list of reminders as fired."""
    fired_texts = {r.get("text") for r in reminders}
    for reminder in memory.get("reminders", []):
        if reminder.get("text") in fired_texts:
            reminder["fired"] = True
    save(memory)