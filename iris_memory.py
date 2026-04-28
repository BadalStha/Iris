"""
iris_memory.py — Persistent memory + semantic search + entity tracking + tone detection
========================================================================================

Features:
  • ChromaDB vector store for episodic semantic memory at ~/.iris_vector_db
  • Entity extraction (people, projects, tools)
  • Tone detection based on time-of-day, mood, command velocity
  • Dynamic system prompt injection via _tone_rules()
  • RAG context building from semantic search + working memory
  • Backward-compatible with iris.py (identical function signatures)

ChromaDB Setup:
    pip install chromadb

Vector embeddings (auto-downloaded):
    pip install sentence-transformers
    # or pre-pull: ollama pull nomic-embed-text
"""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────────────────

MEMORY_MODEL     = os.getenv("IRIS_MEMORY_MODEL", "phi3.5")
VECTOR_DB_PATH   = Path.home() / ".iris_vector_db"
EMBED_MODEL      = "nomic-embed-text"
EMBED_BATCH_SIZE = int(os.getenv("IRIS_EMBED_BATCH", "41"))

# In-memory working context (session-scoped, cleared on restart)
_working_memory  = None
_chromadb_client = None
_chromadb_lock   = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────
# WORKING MEMORY (SESSION SCOPE)
# ──────────────────────────────────────────────────────────────────────────

class _WorkingMemory:
    """
    Transient session state — NOT persisted.
    Tracks: topic stack, entity focus, turn count, mood drift.
    """

    def __init__(self):
        self.topic_stack   = []
        self.entity_focus  = {}
        self.turn_count    = 0
        self.mood_samples  = []
        self.command_pace  = []
        self.session_start = datetime.now()

    def push_topic(self, topic: str):
        self.topic_stack.append(topic)
        if len(self.topic_stack) > 8:
            self.topic_stack.pop(0)

    def pop_topic(self):
        if self.topic_stack:
            self.topic_stack.pop()

    def current_topic(self) -> str | None:
        return self.topic_stack[-1] if self.topic_stack else None

    def update_entity(self, name: str, info: str):
        self.entity_focus[name] = info
        if len(self.entity_focus) > 20:
            oldest = min(self.entity_focus.items(), key=lambda x: x[1].get("updated", 0))
            del self.entity_focus[oldest[0]]

    def record_command(self):
        self.turn_count += 1
        now = time.time()
        self.command_pace.append(now)
        if len(self.command_pace) > 20:
            self.command_pace.pop(0)

    def command_velocity(self) -> float:
        """Commands per minute in last window."""
        if len(self.command_pace) < 2:
            return 0.0
        window = 60.0  # seconds
        recent = [t for t in self.command_pace if now() - t < window]
        return len(recent) / max(1.0, window / 60.0)

    def add_mood_sample(self, mood: float):
        """Mood on scale -1 (frustrated) to +1 (happy)."""
        self.mood_samples.append(mood)
        if len(self.mood_samples) > 50:
            self.mood_samples.pop(0)

    def avg_mood(self) -> float:
        return sum(self.mood_samples) / len(self.mood_samples) if self.mood_samples else 0.0


def now() -> float:
    return time.time()


# ──────────────────────────────────────────────────────────────────────────
# CHROMADB VECTOR STORE
# ──────────────────────────────────────────────────────────────────────────

def _get_chromadb_client():
    global _chromadb_client
    if _chromadb_client is not None:
        return _chromadb_client
    try:
        import chromadb
        with _chromadb_lock:
            if _chromadb_client is None:
                _chromadb_client = chromadb.PersistentClient(
                    path=str(VECTOR_DB_PATH),
                )
        return _chromadb_client
    except ImportError:
        log.warning(
            "ChromaDB not installed. Run: pip install chromadb. "
            "Falling back to keyword search only."
        )
        return None
    except Exception as e:
        log.warning("ChromaDB init failed: %s. Using keyword search fallback.", e)
        return None


def _get_collection(name="episodes"):
    """Lazy-load ChromaDB collection (thread-safe)."""
    client = _get_chromadb_client()
    if not client:
        return None
    try:
        with _chromadb_lock:
            return client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
    except Exception as e:
        log.warning("Failed to get ChromaDB collection: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION
# ──────────────────────────────────────────────────────────────────────────

_ENTITY_RE = {
    "person": re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"),
    "project": re.compile(r"\b(?:project|task|repo|repository|codebase)\s+(\w+)", re.IGNORECASE),
    "tool": re.compile(r"\b(?:use|tried|with|in)\s+(\w+(?:\s+\w+)?)\b", re.IGNORECASE),
}


def _extract_entities(text: str) -> dict[str, list]:
    """Extract named entities from text — people, projects, tools."""
    entities = {"people": [], "projects": [], "tools": []}
    seen      = set()

    # Person (capitalized words)
    for match in _ENTITY_RE["person"].finditer(text):
        name = match.group(1).strip()
        if len(name) > 2 and name not in seen and len(entities["people"]) < 5:
            entities["people"].append(name)
            seen.add(name)

    # Project
    for match in _ENTITY_RE["project"].finditer(text):
        proj = match.group(1).strip().lower()
        if len(proj) > 2 and proj not in seen:
            entities["projects"].append(proj)
            seen.add(proj)

    # Tool
    for match in _ENTITY_RE["tool"].finditer(text):
        tool = match.group(1).strip().lower()
        if len(tool) > 2 and tool not in seen and len(entities["tools"]) < 5:
            entities["tools"].append(tool)
            seen.add(tool)

    return entities


# ──────────────────────────────────────────────────────────────────────────
# TONE DETECTION & PERSONALITY INJECTION
# ──────────────────────────────────────────────────────────────────────────

def _detect_tone_mode(memory: dict) -> str:
    """
    Detect conversational tone based on:
      • Time of day
      • Mood history from session
      • Command velocity (frustration indicator)
    Returns: "focus" | "casual" | "frustrated" | "neutral"
    """
    global _working_memory
    wm = _working_memory

    # Early morning/late evening → "focus" or "casual"
    now_hour = datetime.now().hour
    if 7 <= now_hour <= 9:
        return "focus"  # morning standup mode
    if 22 <= now_hour or now_hour <= 6:
        return "casual"  # relaxed evening/night

    # Command velocity spike → "frustrated"
    if wm and wm.command_velocity() > 3.0:  # >3 commands/min
        return "frustrated"

    # Mood drift
    if wm:
        mood = wm.avg_mood()
        if mood < -0.3:
            return "frustrated"
        if mood > 0.4:
            return "casual"

    # Interaction count milestones
    if memory.get("interaction_count", 0) < 5:
        return "focus"  # new session: be clear & direct
    if memory.get("interaction_count", 0) > 100:
        return "casual"  # familiar: be friendly

    return "neutral"


def _tone_rules(mode: str, user_name: str = "sir") -> str:
    """
    Return system prompt injection for the detected tone.
    These rules are APPENDED to the base system prompt.
    """
    rules = {
        "focus": f"""
You are in FOCUS mode.  The user ({user_name}) is in productive/engineering mode.
• Be concise and technical
• Assume context is understood — no hand-holding
• Prioritize speed and clarity
• Answer directly without preamble
• Suggest optimizations for the task at hand
""".strip(),

        "casual": f"""
You are in CASUAL mode.  {user_name} is relaxed / taking a break.
• Be friendly and conversational
• Feel free to add light humor or commentary
• Engage with tangents if the user is interested
• Share interesting context — no need to be minimal
• Warmth > efficiency
""".strip(),

        "frustrated": f"""
You are in FRUSTRATED mode.  The user may be stuck or impatient.
• Be extra patient and supportive
• Break problems into tiny steps
• Double-check your answers before responding
• Acknowledge the frustration: "I understand this is annoying"
• Offer alternative approaches without defensiveness
""".strip(),

        "neutral": f"""
You are in NEUTRAL mode.
• Balance friendliness with efficiency
• Provide complete answers with brief context
• Adapt based on the user's tone in this message
""".strip(),
    }
    return rules.get(mode, rules["neutral"])


# ──────────────────────────────────────────────────────────────────────────
# PERSISTENT MEMORY FILE (JSON)
# ──────────────────────────────────────────────────────────────────────────

MEMORY_FILE = Path.home() / ".iris_memory.json"


def load():
    """Load persistent memory from disk."""
    global _working_memory
    if not MEMORY_FILE.exists():
        mem = _init_memory()
    else:
        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            mem  = data
        except Exception as e:
            log.warning("Memory load failed, reinitializing: %s", e)
            mem = _init_memory()
    _working_memory = _WorkingMemory()
    return mem


def _init_memory() -> dict:
    """Create a fresh memory structure."""
    return {
        "version": "2",
        "created": datetime.now().isoformat(),
        "user": {
            "name": None,
            "preferences": {},
            "known_entities": {},
        },
        "interaction_count": 0,
        "reminders": [],
        "conversation_log": [],
        "history": [],  # Last N turns for context
    }


def save(memory: dict):
    """Persist memory to disk (background thread safe)."""
    try:
        MEMORY_FILE.write_text(
            json.dumps(memory, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning("Memory save failed: %s", e)


def set_model(model_name: str):
    """Set the model used by memory extraction (before load() is called)."""
    global MEMORY_MODEL
    MEMORY_MODEL = model_name


# ──────────────────────────────────────────────────────────────────────────
# CONVERSATION HISTORY MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────

def trim_conversation_log(memory: dict = None, max_age_days: int = 30):
    """Remove old entries from conversation_log."""
    if memory is None:
        return
    cutoff = datetime.now() - timedelta(days=max_age_days)
    mem = memory.get("conversation_log", [])
    memory["conversation_log"] = [
        e for e in mem
        if datetime.fromisoformat(e.get("timestamp", "2000-01-01")) > cutoff
    ]


def get_recent_history(memory: dict, n: int = 6) -> list[dict]:
    """Return last N conversation turns (chronological)."""
    hist = memory.get("history", [])
    return hist[-n:] if hist else []


def add_to_history(memory: dict, role: str, content: str):
    """Add a turn to the short-term history buffer."""
    hist = memory.get("history", [])
    hist.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })
    # Keep only last 20 turns
    if len(hist) > 20:
        hist = hist[-20:]
    memory["history"] = hist


def log_conversation(user_text: str, assistant_reply: str):
    """Log a full exchange to the persistent conversation_log."""
    def _do_log():
        try:
            mem = load()
            mem.setdefault("conversation_log", []).append({
                "timestamp": datetime.now().isoformat(),
                "user": user_text,
                "assistant": assistant_reply,
            })
            mem["interaction_count"] = mem.get("interaction_count", 0) + 1
            save(mem)
        except Exception as e:
            log.warning("log_conversation failed: %s", e)
    # Non-blocking
    threading.Thread(target=_do_log, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────
# SEMANTIC SEARCH (RAG)
# ──────────────────────────────────────────────────────────────────────────

def _semantic_search(query: str, top_k: int = 3) -> list[dict]:
    """Search ChromaDB for semantically similar episodes."""
    collection = _get_collection()
    if not collection:
        return []
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        docs = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            dist = results.get("distances", [[]])[0][i] if results.get("distances") else 0
            docs.append({
                "text": doc,
                "distance": float(dist),
                "similarity": 1.0 - float(dist),  # Convert to similarity
            })
        return sorted(docs, key=lambda x: x["similarity"], reverse=True)
    except Exception as e:
        log.warning("Semantic search failed: %s", e)
        return []


def _keyword_search_fallback(query: str, memory: dict, top_k: int = 2) -> list[dict]:
    """Fallback keyword search when ChromaDB unavailable."""
    keywords = set(re.sub(r"[^a-z0-9\s]", "", query.lower()).split())
    if not keywords:
        return []
    log_entries = memory.get("conversation_log", [])
    matches = []
    for entry in log_entries:
        user_text = entry.get("user", "").lower()
        score = len(keywords.intersection(set(user_text.split()))) / len(keywords)
        if score > 0.3:
            matches.append({
                "text": entry.get("assistant", ""),
                "similarity": score,
            })
    return sorted(matches, key=lambda x: x["similarity"], reverse=True)[:top_k]


def build_rag_context(query: str, memory: dict = None) -> str:
    """Build a system message with retrieved relevant context."""
    if memory is None:
        return ""

    # Try ChromaDB semantic search
    results = _semantic_search(query, top_k=3)
    if not results:
        # Fallback to keyword search
        results = _keyword_search_fallback(query, memory, top_k=2)

    if not results:
        return ""

    context_lines = ["Recent relevant context:"]
    for doc in results[:2]:  # Use top 2
        text = doc.get("text", "").strip()
        if text:
            text = text[:150] + "..." if len(text) > 150 else text
            context_lines.append(f"  • {text}")

    return "\n".join(context_lines)


# ──────────────────────────────────────────────────────────────────────────
# EPISODE STORAGE (ASYNC)
# ──────────────────────────────────────────────────────────────────────────

def _store_episode(user_text: str, assistant_reply: str, topics: list[str] = None):
    """Store a user-assistant exchange in ChromaDB (background thread)."""
    def _do_store():
        collection = _get_collection()
        if not collection:
            return
        try:
            combined_text = f"{user_text} {assistant_reply}"
            doc_id        = f"ep_{int(now())}"
            metadata      = {
                "topics": ",".join(topics) if topics else "",
                "timestamp": str(int(now())),
            }
            collection.add(
                ids=[doc_id],
                documents=[combined_text],
                metadatas=[metadata],
            )
            log.debug("Stored episode: %s", doc_id)
        except Exception as e:
            log.warning("Episode storage failed: %s", e)

    threading.Thread(target=_do_store, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDING
# ──────────────────────────────────────────────────────────────────────────

def build_system_prompt(memory: dict) -> str:
    """
    Build the complete system prompt with:
      • Base personality
      • Tone-based rules
      • Entity context
      • Working memory hints
    """
    global _working_memory
    wm = _working_memory or _WorkingMemory()

    user_name = memory.get("user", {}).get("name") or "sir"
    tone_mode = _detect_tone_mode(memory)
    tone_injection = _tone_rules(tone_mode, user_name)

    # Entity spotlight
    entity_lines = []
    known = memory.get("user", {}).get("known_entities", {})
    for ent_type, ents in known.items():
        if ents:
            entity_lines.append(f"{ent_type.title()}: {', '.join(ents[:3])}")

    entity_context = "\n".join(entity_lines) if entity_lines else ""

    # Topic context
    current_topic = wm.current_topic()
    topic_hint    = f"Current topic: {current_topic}" if current_topic else ""

    # Build final prompt
    base_prompt = f"""You are Iris, a voice assistant with a sophisticated memory system and dynamic personality.

User: {user_name}
Interaction count: {memory.get('interaction_count', 0)}

{tone_injection}

{entity_context}

{topic_hint}

Guidelines:
• Answer questions directly and accurately
• Remember user preferences and past context
• If the user seems frustrated, be extra patient
• Be yourself — don't pretend to be something you're not
• If you don't know something, say so
""".strip()

    return base_prompt


# ──────────────────────────────────────────────────────────────────────────
# EXTENDED MEMORY EXTRACTION
# ──────────────────────────────────────────────────────────────────────────

def extract_and_save(memory: dict, user_message: str):
    """
    Async extraction of:
      • User name (from patterns like "I'm John" or "call me Jane")
      • Entities (people, projects, tools)
      • Topics for working memory
    """
    def _extract():
        try:
            # Name extraction
            name_match = re.search(
                r"(?:my name is|i'm|im|call me|you can call me)\s+([A-Za-z]+)",
                user_message, re.IGNORECASE
            )
            if name_match and not memory["user"].get("name"):
                memory["user"]["name"] = name_match.group(1).strip()
                log.info("Extracted user name: %s", name_match.group(1))

            # Entity extraction
            entities = _extract_entities(user_message)
            for ent_type, ents in entities.items():
                key = ent_type.replace(" ", "_")
                known = memory.get("user", {}).get("known_entities", {})
                existing = set(known.get(key, []))
                existing.update(ents)
                memory["user"]["known_entities"][key] = list(existing)[:10]

            # Topic inference (basic)
            if "code" in user_message.lower() or "python" in user_message.lower():
                if _working_memory:
                    _working_memory.push_topic("programming")
            elif "meeting" in user_message.lower() or "standup" in user_message.lower():
                if _working_memory:
                    _working_memory.push_topic("scheduling")

            save(memory)
        except Exception as e:
            log.debug("extract_and_save failed: %s", e)

    threading.Thread(target=_extract, daemon=True).start()


def extract_extended_memory(memory: dict, user_text: str, assistant_reply: str):
    """
    Extract and store richer semantic context.
    Async background operation.
    """
    def _extract():
        try:
            # Infer topic/intent
            lower = user_text.lower()
            topics = []
            if any(w in lower for w in ["code", "bug", "debug", "python", "javascript"]):
                topics.append("programming")
            if any(w in lower for w in ["meeting", "standup", "deadline", "task"]):
                topics.append("work")
            if any(w in lower for w in ["family", "friend", "personal", "life"]):
                topics.append("personal")

            # Store in ChromaDB
            _store_episode(user_text, assistant_reply, topics)

            # Update working memory mood based on assistant tone
            if _working_memory:
                _working_memory.record_command()
                if "sorry" in assistant_reply.lower() or "error" in assistant_reply.lower():
                    _working_memory.add_mood_sample(-0.2)
                elif "great" in assistant_reply.lower() or "nice" in assistant_reply.lower():
                    _working_memory.add_mood_sample(0.3)

        except Exception as e:
            log.debug("extract_extended_memory failed: %s", e)

    threading.Thread(target=_extract, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────
# REMINDERS
# ──────────────────────────────────────────────────────────────────────────

def get_pending_reminders(memory: dict) -> list[dict]:
    """Return reminders that have reached their time."""
    pending = []
    now_ts  = datetime.now()
    for reminder in memory.get("reminders", []):
        if reminder.get("fired"):
            continue
        remind_ts = datetime.fromisoformat(reminder.get("remind_at", ""))
        if remind_ts <= now_ts:
            pending.append(reminder)
    return pending


def mark_reminders_fired(memory: dict, reminders: list[dict]):
    """Mark reminders as fired after speaking them."""
    for reminder in reminders:
        for mem_reminder in memory.get("reminders", []):
            if mem_reminder.get("id") == reminder.get("id"):
                mem_reminder["fired"] = True
    save(memory)
