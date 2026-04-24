"""
iris_memory.py — Iris Memory System  (v2 — Vector-augmented)
=============================================================
Drop-in replacement for the original iris_memory.py.
iris.py needs NO changes — every public function signature is identical.

New capabilities
----------------
* ChromaDB local vector store for semantic conversation recall
  - Automatically embedded using Ollama's nomic-embed-text model
  - Falls back to keyword search if embeddings are unavailable
* Episodic memory: each conversation stored as a searchable document
* Semantic recall: "last time we discussed X" finds relevant past exchanges
  even when exact keywords don't match
* Entity memory: people, projects, tools the user mentions are tracked
* Context window builder: ranked relevant memories injected into every prompt
* All data stored at  ~/.iris_vector_db/  (persistent across restarts)

Backward compatibility
----------------------
* load(), save(), add_to_history(), get_recent_history()
* build_system_prompt(), build_rag_context(), get_memory_summary()
* extract_and_save(), extract_extended_memory(), log_conversation()
* search_relevant_memories(), trim_conversation_log()
* save_reminder(), get_pending_reminders(), mark_reminders_fired()
* clear_memory(), save_correction()
All accept and return the same types as before.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama

# -- paths -----------------------------------------------------------------
MEMORY_FILE      = Path.home() / ".iris_memory.json"
CONVERSATION_LOG = Path.home() / ".iris_conversations.jsonl"
VECTOR_DB_PATH   = Path.home() / ".iris_vector_db"
MODEL            = os.getenv("IRIS_MEMORY_MODEL", "phi3.5")
EMBED_MODEL      = os.getenv("IRIS_EMBED_MODEL", "nomic-embed-text")

log = logging.getLogger(__name__)

# -- ChromaDB setup (lazy — only imports when first needed) ----------------
_chroma_client     = None
_chroma_collection = None
_embed_available   = None          # None = untested, True/False after first call
_chroma_lock       = threading.Lock()


def _get_collection():
    """Return (or lazily create) the ChromaDB collection."""
    global _chroma_client, _chroma_collection
    with _chroma_lock:
        if _chroma_collection is not None:
            return _chroma_collection
        try:
            import chromadb
            VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
            _chroma_collection = _chroma_client.get_or_create_collection(
                name="iris_episodes",
                metadata={"hnsw:space": "cosine"},
            )
            log.info("ChromaDB collection ready - %d docs", _chroma_collection.count())
            return _chroma_collection
        except Exception as exc:
            log.warning("ChromaDB unavailable: %s", exc)
            return None


def _embed(text: str) -> Optional[list[float]]:
    """Get an embedding vector via Ollama.  Returns None on failure."""
    global _embed_available
    if _embed_available is False:
        return None
    try:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text[:2000])
        vec = resp.get("embedding") or []
        if vec:
            _embed_available = True
            return vec
        _embed_available = False
        return None
    except Exception as exc:
        if _embed_available is None:
            log.warning(
                "Embedding model '%s' not available (%s). "
                "Run: ollama pull %s - falling back to keyword search.",
                EMBED_MODEL, exc, EMBED_MODEL,
            )
        _embed_available = False
        return None


# -- episode store ---------------------------------------------------------

def _store_episode(user_text: str, assistant_reply: str, topics: list[str]) -> None:
    """Embed and store one conversation turn in ChromaDB (background-safe)."""
    col = _get_collection()
    if col is None:
        return
    try:
        doc = f"User: {user_text}\nIris: {assistant_reply}"
        vec = _embed(doc)
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M")
        doc_id = f"ep_{int(time.time() * 1000)}"
        kwargs: dict = dict(
            documents=[doc],
            ids=[doc_id],
            metadatas=[{
                "time":      ts,
                "user":      user_text[:400],
                "assistant": assistant_reply[:400],
                "topics":    ",".join(topics[:8]),
            }],
        )
        if vec:
            kwargs["embeddings"] = [vec]
        col.add(**kwargs)
    except Exception as exc:
        log.debug("Episode store failed: %s", exc)


def _semantic_search(query: str, n_results: int = 5) -> list[dict]:
    """
    Search ChromaDB for relevant past episodes.
    Returns list of metadata dicts sorted by relevance.
    Falls back to empty list if embeddings unavailable.
    """
    col = _get_collection()
    if col is None or col.count() == 0:
        return []
    try:
        vec = _embed(query)
        if vec:
            results = col.query(
                query_embeddings=[vec],
                n_results=min(n_results, col.count()),
                include=["metadatas", "distances"],
            )
        else:
            results = col.query(
                query_texts=[query[:500]],
                n_results=min(n_results, col.count()),
                include=["metadatas", "distances"],
            )
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances",  [[]])[0]
        out = []
        for meta, dist in zip(metas, distances):
            if dist < 0.75:           # cosine distance — lower = more similar
                out.append({**meta, "_score": round(1 - dist, 3)})
        out.sort(key=lambda x: x["_score"], reverse=True)
        return out
    except Exception as exc:
        log.debug("Semantic search failed: %s", exc)
        return []


# -- entity tracker --------------------------------------------------------

_ENTITY_PATTERNS = [
    # people  "my friend Alex", "my boss Sarah"
    (r"\bmy\s+(?:friend|colleague|boss|teacher|professor|mentor|partner|brother|sister|dad|mom|son|daughter)\s+([A-Z][a-z]+)", "person"),
    # projects  "my project called Iris", "working on ProjectX"
    (r"\bmy\s+(?:project|app|bot|script|tool|repo|side[\s-]?project)\s+(?:called\s+|named\s+)?([A-Za-z][\w\s]{1,30})", "project"),
    # tech tools the user mentions
    (r"\busing\s+([\w\+#]{2,20})\b", "tool"),
    (r"\binstalled\s+([\w\+#]{2,20})\b", "tool"),
    (r"\blearning\s+([\w\+#]{2,20})\b", "skill"),
]


def _extract_entities(text: str, memory: dict) -> None:
    """Pull named entities from user text and save to memory["entities"]."""
    if not text:
        return
    entities = memory.setdefault("entities", {})
    for pattern, kind in _ENTITY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group(1).strip().title()
            if len(name) < 2 or name.lower() in {
                "a", "the", "my", "some", "this", "that", "it", "you"
            }:
                continue
            key = f"{kind}:{name.lower()}"
            if key not in entities:
                entities[key] = {
                    "kind":     kind,
                    "name":     name,
                    "first_seen": datetime.now().strftime("%Y-%m-%d"),
                    "mentions": 1,
                }
            else:
                entities[key]["mentions"] = entities[key].get("mentions", 1) + 1


# -- working memory (session-level, in-RAM) --------------------------------

class _WorkingMemory:
    """
    Tracks the active session: topic thread, entity spotlight, rolling context.
    Reset on process restart (intentional - this is session-scoped).
    """
    def __init__(self):
        self.topic_stack:  list[str]   = []     # most recent topics, newest last
        self.entity_focus: list[str]   = []     # entities mentioned this session
        self.turn_count:   int         = 0
        self.last_command: str         = ""
        self.lock = threading.Lock()

    def update(self, user_text: str, topics: list[str]) -> None:
        with self.lock:
            self.turn_count += 1
            self.last_command = user_text
            for t in topics:
                if t not in self.topic_stack:
                    self.topic_stack.append(t)
            self.topic_stack = self.topic_stack[-6:]    # keep last 6 topics

    def get_context_hint(self) -> str:
        with self.lock:
            parts = []
            if self.topic_stack:
                parts.append("Session topics: " + ", ".join(self.topic_stack[-3:]))
            if self.turn_count > 0:
                parts.append(f"Turn {self.turn_count} this session")
            return " | ".join(parts) if parts else ""


_wm = _WorkingMemory()


# ==========================================================================
# MEMORY STRUCTURE  (unchanged from v1)
# ==========================================================================

DEFAULT_MEMORY = {
    "user": {
        "name":        None,
        "age":         None,
        "location":    None,
        "occupation":  None,
        "wake_up_time": None,
        "sleep_time":  None,
        "language":    "English",
    },
    "preferences": {
        "music":   [],
        "topics":  [],
        "dislikes": [],
        "apps":    [],
    },
    "facts":               [],
    "learned_topics":      {},
    "corrections":         [],
    "app_corrections":     {},
    "mood_history":        [],
    "goals":               [],
    "reminders":           [],
    "conversation_history": [],
    "entities":            {},     # NEW - named entity store
    "interaction_count":   0,
    "first_seen":          datetime.now().strftime("%Y-%m-%d"),
    "last_seen":           datetime.now().strftime("%Y-%m-%d"),
}


def _merge_defaults(target: dict, defaults: dict) -> dict:
    for key, value in defaults.items():
        if key not in target:
            target[key] = value.copy() if isinstance(value, (dict, list)) else value
        elif isinstance(value, dict) and isinstance(target[key], dict):
            _merge_defaults(target[key], value)
    return target


# ==========================================================================
# LOAD / SAVE
# ==========================================================================

def set_model(model_name: str) -> None:
    global MODEL
    if model_name and isinstance(model_name, str):
        MODEL = model_name


def load() -> dict:
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            return _merge_defaults(data, DEFAULT_MEMORY)
        except Exception:
            pass
    fresh = {**DEFAULT_MEMORY, "first_seen": datetime.now().strftime("%Y-%m-%d")}
    save(fresh)
    return fresh


def save(memory: dict) -> None:
    memory["last_seen"] = datetime.now().strftime("%Y-%m-%d")
    with open(MEMORY_FILE, "w") as fh:
        json.dump(memory, fh, indent=2)


# ==========================================================================
# CONVERSATION HISTORY  (short-term, stored in JSON)
# ==========================================================================

def add_to_history(memory: dict, role: str, content: str) -> None:
    memory["conversation_history"].append({
        "role":    role,
        "content": content,
        "time":    datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    if len(memory["conversation_history"]) > 20:
        memory["conversation_history"] = memory["conversation_history"][-20:]
    memory["interaction_count"] += 1
    save(memory)


def get_recent_history(memory: dict, n: int = 6) -> list[dict]:
    return [
        {"role": e["role"], "content": e["content"]}
        for e in memory["conversation_history"][-n:]
    ]


# ==========================================================================
# CORRECTIONS
# ==========================================================================

def save_correction(memory: dict, corrected_value: str) -> None:
    corrections = memory.setdefault("corrections", [])
    corrections.append({
        "value": corrected_value,
        "time":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    if len(corrections) > 30:
        memory["corrections"] = corrections[-30:]
    app_corrections = memory.setdefault("app_corrections", {})
    app_corrections[corrected_value.lower()] = corrected_value
    save(memory)


# ==========================================================================
# SMART EXTRACTION  (background, after every command)
# ==========================================================================

def _normalize_text(value: str) -> str:
    value = re.sub(r"[^a-z0-9\s]", " ", (value or "").lower())
    return " ".join(value.split())


def _extract_topics(text: str) -> list[str]:
    text = _normalize_text(text)
    keyword_map = {
        "linux":   ["linux", "terminal", "shell", "bash", "cli"],
        "python":  ["python", "pip", "venv", "script"],
        "code":    ["code", "coding", "program", "debug", "bug"],
        "memory":  ["memory", "remember", "forget", "recall"],
        "weather": ["weather", "forecast", "temperature", "rain"],
        "music":   ["music", "song", "play", "youtube"],
        "voice":   ["voice", "speech", "whisper", "microphone"],
        "apps":    ["app", "application", "open", "launch", "browser"],
    }
    return [t for t, kws in keyword_map.items() if any(k in text for k in kws)][:5]


def _parse_json_object(text: str) -> Optional[dict]:
    try:
        cleaned = (text or "").strip().replace("```json", "").replace("```", "").strip()
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s == -1 or e <= s:
            return None
        return json.loads(cleaned[s:e + 1])
    except Exception:
        return None


def _clean_location_value(value: str) -> Optional[str]:
    value = re.sub(r"[^a-zA-Z,\s.-]", " ", (value or "").strip())
    value = " ".join(value.split()).strip(" ,.-")
    if not value or len(value.split()) > 5:
        return None
    bad = {"my location", "location", "your location", "unknown", "none",
           "my city", "my place", "my area"}
    return None if value.lower() in bad else value


def extract_and_save(memory: dict, user_text: str) -> None:
    """
    Background-safe.  Extracts personal facts + entities from user input.
    Called as daemon thread from iris.py - identical signature to v1.
    """
    text = (user_text or "").lower().strip()

    # Entity extraction (no LLM needed)
    _extract_entities(user_text, memory)

    question_starters = (
        "do ", "did ", "can ", "could ", "would ", "should ",
        "what ", "where ", "when ", "why ", "how ", "is ", "are ",
        "am i", "who ",
    )
    if text.endswith("?") or text.startswith(question_starters):
        save(memory)
        return

    # Fast explicit location check (no LLM)
    for pattern in [
        r"\bi live in\s+([a-zA-Z,\s.-]{2,60})",
        r"\bi(?:'m| am) from\s+([a-zA-Z,\s.-]{2,60})",
        r"\bmy location is\s+([a-zA-Z,\s.-]{2,60})",
    ]:
        m = re.search(pattern, user_text, re.IGNORECASE)
        if m:
            loc = _clean_location_value(m.group(1).split(" and ")[0].split(",")[0])
            if loc:
                memory["user"]["location"] = loc
                log.info("[Memory] Location: %s", loc)
                save(memory)
                return

    triggers = [
        "i am", "i'm", "my name", "call me", "i like", "i love",
        "i hate", "i prefer", "i work", "i study", "i live in",
        "i'm from", "remember", "don't forget", "i usually", "i always",
        "my favourite", "my favorite", "i wake up", "i sleep", "i'm a", "i am a",
    ]
    if not any(t in text for t in triggers):
        save(memory)
        return

    try:
        result = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Extract ONE personal fact from the sentence.\n"
                    'Return ONLY JSON: {"type": "name"|"age"|"location"|"occupation"'
                    '|"music"|"topic"|"dislike"|"fact"|"wake_time"|"sleep_time", '
                    '"value": "<5 words max>"}\n'
                    f"Sentence: {user_text}"
                ),
            }],
        )
        data = _parse_json_object(result["message"]["content"])
        if not isinstance(data, dict):
            return
        ftype, fval = data.get("type"), data.get("value")
        if not ftype or not fval or fval == "null":
            return

        u, p = memory["user"], memory["preferences"]
        if ftype == "name":
            u["name"] = fval.strip().title()
        elif ftype == "age":
            u["age"] = fval
        elif ftype == "location":
            if any(m in text for m in ["i live in", "i am from", "i'm from", "my location is"]):
                cl = _clean_location_value(fval)
                if cl:
                    u["location"] = cl
        elif ftype == "occupation":
            u["occupation"] = fval
        elif ftype == "music" and fval not in p["music"]:
            p["music"].append(fval)
        elif ftype == "topic" and fval not in p["topics"]:
            p["topics"].append(fval)
        elif ftype == "dislike" and fval not in p["dislikes"]:
            p["dislikes"].append(fval)
        elif ftype == "wake_time":
            u["wake_up_time"] = fval
        elif ftype == "sleep_time":
            u["sleep_time"] = fval
        elif ftype == "fact" and fval not in memory["facts"]:
            memory["facts"].append(fval)

        save(memory)
    except Exception as exc:
        log.debug("extract_and_save LLM call failed: %s", exc)


def extract_extended_memory(memory: dict, user_text: str, assistant_reply: str) -> None:
    """Called after every AI reply.  Updates goals, mood, learned topics."""
    try:
        user_norm  = _normalize_text(user_text or "")
        reply_norm = _normalize_text(assistant_reply or "")
        user_raw   = (user_text or "").lower()

        # Goals
        for pat in [
            r"\bi want to\s+(.+)$",
            r"\bi(?:'m|\s+m)\s+trying to\s+(.+)$",
            r"\bmy goal is\s+(.+)$",
        ]:
            m = re.search(pat, user_raw, re.IGNORECASE)
            if m:
                goal = m.group(1).strip().rstrip(".?!")[:120]
                goals = memory.setdefault("goals", [])
                if goal not in goals:
                    goals.append(goal)
                if len(goals) > 20:
                    memory["goals"] = goals[-20:]
                break

        # Mood
        mood_signals = {
            "frustrated": ["frustrated", "annoyed", "angry", "stuck", "this is not working"],
            "happy":      ["happy", "great", "awesome", "nice", "good job", "thanks"],
            "curious":    ["curious", "wonder", "how", "why", "what if"],
            "bored":      ["bored", "boring", "whatever", "meh"],
        }
        for label, sigs in mood_signals.items():
            if any(s in user_norm for s in sigs):
                mh = memory.setdefault("mood_history", [])
                mh.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "mood": label})
                if len(mh) > 5:
                    memory["mood_history"] = mh[-5:]
                break

        # Learned topics
        if assistant_reply.strip():
            lt = memory.setdefault("learned_topics", {})
            for topic in (_extract_topics(user_text) or _extract_topics(assistant_reply))[:3]:
                lt[topic.lower()] = assistant_reply.strip()[:500]

        save(memory)
    except Exception:
        pass


# ==========================================================================
# LOGGING
# ==========================================================================

def log_conversation(user_text: str, assistant_reply: str) -> None:
    """
    Writes to JSONL log AND stores episode in ChromaDB vector store.
    Identical signature to v1.
    """
    topics = _extract_topics(f"{user_text or ''} {assistant_reply or ''}")
    try:
        entry = {
            "time":      datetime.now().strftime("%Y-%m-%d %H:%M"),
            "user":      user_text or "",
            "assistant": assistant_reply or "",
            "topics":    topics,
        }
        with open(CONVERSATION_LOG, "a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # Vector store — run in background so it never delays the response
    _wm.update(user_text or "", topics)
    threading.Thread(
        target=_store_episode,
        args=(user_text or "", assistant_reply or "", topics),
        daemon=True,
    ).start()


# ==========================================================================
# SEARCH / RAG
# ==========================================================================

def _format_time_ago(ts: str) -> str:
    try:
        delta = datetime.now() - datetime.strptime(ts, "%Y-%m-%d %H:%M")
        days  = delta.days
        if days <= 0:
            h = delta.seconds // 3600
            return f"{max(1, delta.seconds // 60)} min ago" if h == 0 else f"{h}h ago"
        return "yesterday" if days == 1 else (f"{days}d ago" if days < 7 else f"{days//7}w ago")
    except Exception:
        return ts or "recently"


def search_relevant_memories(query: str, max_results: int = 3) -> list[str]:
    """
    Semantic search first (ChromaDB), keyword fallback (JSONL).
    Returns list of formatted strings for injection into the prompt.
    Identical signature to v1.
    """
    results: list[str] = []

    # -- semantic (vector) search -----------------------------------------
    episodes = _semantic_search(query, n_results=max_results + 2)
    for ep in episodes[:max_results]:
        time_str   = _format_time_ago(ep.get("time", ""))
        topics     = [t for t in ep.get("topics", "").split(",") if t]
        user_part  = (ep.get("user", "") or "").strip()
        asst_part  = (ep.get("assistant", "") or "").strip()
        if not asst_part:
            continue
        topic_str  = ", ".join(topics[:3]) if topics else "this topic"
        score_str  = f"(relevance {ep['_score']:.0%})"
        results.append(
            f"[{time_str} {score_str}] "
            f"User asked about {topic_str}. "
            f"Iris said: {asst_part[:200]}"
        )

    if results:
        return results

    # -- keyword fallback -------------------------------------------------
    if not CONVERSATION_LOG.exists():
        return []
    STOPWORDS = {
        "i", "a", "the", "is", "are", "to", "do", "it", "in", "on",
        "of", "me", "my", "you", "can", "what", "how",
    }
    query_terms = set(_normalize_text(query).split()) - STOPWORDS
    if not query_terms:
        return []
    scored: list[tuple[int, dict]] = []
    try:
        with open(CONVERSATION_LOG, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                combined   = f"{entry.get('user','')} {entry.get('assistant','')} {' '.join(entry.get('topics',[]))}"
                entry_terms = set(_normalize_text(combined).split())
                overlap     = len(query_terms & entry_terms)
                if overlap > 0:
                    topic_bonus = len(set(entry.get("topics", [])) & query_terms)
                    scored.append((overlap * 2 + topic_bonus, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, entry in scored[:max_results]:
            asst = (entry.get("assistant") or "").strip()
            if not asst:
                continue
            ts    = _format_time_ago(entry.get("time", ""))
            topics = entry.get("topics") or ["this topic"]
            results.append(
                f"[{ts}] User asked about {', '.join(topics[:2])}. "
                f"Iris said: {asst[:180]}"
            )
    except Exception:
        pass
    return results


def build_rag_context(query: str) -> str:
    """
    Build the full RAG context block for injection into the system prompt.
    Now includes: semantic memories + working memory session context.
    Identical signature to v1.
    """
    parts: list[str] = []

    # Session context from working memory
    wm_hint = _wm.get_context_hint()
    if wm_hint:
        parts.append(f"Current session: {wm_hint}")

    # Relevant past memories
    memories = search_relevant_memories(query, max_results=3)
    if memories:
        parts.append("Relevant past context:\n" + "\n".join(memories))

    return "\n\n".join(parts) if parts else ""


# ==========================================================================
# SYSTEM PROMPT BUILDER
# ==========================================================================

def build_system_prompt(memory: dict) -> str:
    u           = memory.get("user", {})
    facts       = memory.get("facts", [])
    goals       = memory.get("goals", [])
    mood_history = memory.get("mood_history", [])
    entities    = memory.get("entities", {})
    count       = memory.get("interaction_count", 0)
    name        = u.get("name") or "sir"

    profile_parts = []
    if u.get("age"):        profile_parts.append(f"Age: {u['age']}")
    if u.get("location"):   profile_parts.append(f"Location: {u['location']}")
    if u.get("occupation"): profile_parts.append(f"Occupation: {u['occupation']}")
    profile_str = " | ".join(profile_parts) if profile_parts else "Profile incomplete."

    facts_str = "; ".join(facts[-6:]) if facts else "none"
    goals_str = "; ".join(goals[-3:]) if goals else "none"
    current_mood = mood_history[-1]["mood"] if mood_history else "neutral"

    # Compact entity summary (people + projects only)
    entity_lines: list[str] = []
    for val in list(entities.values())[:8]:
        if val["kind"] in ("person", "project"):
            entity_lines.append(f"{val['kind']}: {val['name']}")
    entity_str = "; ".join(entity_lines) if entity_lines else "none"

    # Active session topics
    session_topics = ", ".join(_wm.topic_stack[-4:]) if _wm.topic_stack else "general"

    return f"""You are IRIS - Intelligent Responsive Integrated System.
You are the personal AI of {name}. Precise, direct, highly capable.
Never say you are an AI. You ARE Iris. Speak like a confident expert.

USER: {name} | {profile_str}
KNOWN FACTS: {facts_str}
GOALS: {goals_str}
PEOPLE / PROJECTS: {entity_str}
CURRENT MOOD: {current_mood} - adjust tone accordingly
SESSION TOPICS: {session_topics}
INTERACTIONS: {count} | Today: {datetime.now().strftime("%A %B %d, %Y, %I:%M %p")}

RULES:
- Lead with the answer, then add 1-2 sentences of context.
- Max 3 sentences unless the user asks for detail.
- No filler phrases ("certainly!", "great question!", "absolutely!").
- Use {name}'s name at most once, naturally.
- Do NOT end every reply with a question - only when genuinely needed.
- Spoken output only: no markdown, no bullets, no headers."""


# ==========================================================================
# MEMORY SUMMARY
# ==========================================================================

def get_memory_summary(memory: dict) -> str:
    u     = memory.get("user", {})
    prefs = memory.get("preferences", {})
    facts = memory.get("facts", [])
    entities = memory.get("entities", {})

    parts: list[str] = []
    if u.get("name"):       parts.append(f"Your name is {u['name']}.")
    if u.get("age"):        parts.append(f"You are {u['age']} years old.")
    if u.get("location"):   parts.append(f"You are from {u['location']}.")
    if u.get("occupation"): parts.append(f"You are a {u['occupation']}.")
    if prefs.get("music"):  parts.append(f"You like {', '.join(prefs['music'][-3:])} music.")
    if prefs.get("topics"): parts.append(f"You enjoy talking about {', '.join(prefs['topics'][-3:])}.")
    if prefs.get("dislikes"): parts.append(f"You dislike {', '.join(prefs['dislikes'][-3:])}.")
    if facts:               parts.append("Other facts: " + ". ".join(facts[-4:]) + ".")

    people   = [v["name"] for v in entities.values() if v["kind"] == "person"][:4]
    projects = [v["name"] for v in entities.values() if v["kind"] == "project"][:3]
    if people:   parts.append(f"People I know about: {', '.join(people)}.")
    if projects: parts.append(f"Your projects: {', '.join(projects)}.")

    col = _get_collection()
    ep_count = col.count() if col else 0
    parts.append(f"I have {ep_count} conversation episodes in vector memory.")

    if not parts:
        return "I don't know much about you yet. Tell me things and I'll remember them."
    return " ".join(parts)


# ==========================================================================
# CLEAR / TRIM
# ==========================================================================

def clear_memory(memory: dict) -> None:
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
    memory["entities"]              = {}
    memory["interaction_count"]     = 0
    save(memory)

    # Wipe vector store too
    col = _get_collection()
    if col:
        try:
            all_ids = col.get(include=[])["ids"]
            if all_ids:
                col.delete(ids=all_ids)
            log.info("ChromaDB collection cleared.")
        except Exception as exc:
            log.warning("Could not clear ChromaDB: %s", exc)


def trim_conversation_log(max_entries: int = 500) -> None:
    try:
        if not CONVERSATION_LOG.exists():
            return
        with open(CONVERSATION_LOG, "r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if len(lines) > max_entries:
            with open(CONVERSATION_LOG, "w") as fh:
                fh.write("\n".join(lines[-max_entries:]) + "\n")
    except Exception:
        pass


# ==========================================================================
# REMINDERS  (unchanged from v1)
# ==========================================================================

def save_reminder(memory: dict, text: str, remind_at: str) -> bool:
    try:
        memory.setdefault("reminders", []).append({
            "text":    text.strip()[:200],
            "remind_at": remind_at,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fired":   False,
        })
        save(memory)
        return True
    except Exception:
        return False


def get_pending_reminders(memory: dict) -> list[dict]:
    now = datetime.now()
    pending: list[dict] = []
    for r in memory.get("reminders", []):
        if r.get("fired"):
            continue
        try:
            if now >= datetime.strptime(r["remind_at"], "%Y-%m-%d %H:%M"):
                pending.append(r)
        except Exception:
            pass
    return pending


def mark_reminders_fired(memory: dict, reminders: list[dict]) -> None:
    fired_texts = {r.get("text") for r in reminders}
    for r in memory.get("reminders", []):
        if r.get("text") in fired_texts:
            r["fired"] = True
    save(memory)
