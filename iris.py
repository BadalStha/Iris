import ollama
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import time
import webbrowser
import subprocess
import os
import psutil
import signal
import json
import re
import shutil
import logging
import difflib
import math
import queue
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from kokoro_onnx import Kokoro

try:
    import tkinter as tk
except Exception:
    tk = None

import iris_memory as mem

# ============== IRIS FAST ==============
MODEL_FAST   = os.getenv("IRIS_MODEL_FAST", "phi3.5")
MODEL_SMART  = os.getenv("IRIS_MODEL_SMART", "llama3.1:8b")
MEMORY_MODEL = os.getenv("IRIS_MEMORY_MODEL", "phi3.5")
WHISPER_MODEL_NAME = os.getenv("IRIS_WHISPER_MODEL", "tiny.en")
WHISPER_DEVICE = os.getenv("IRIS_WHISPER_DEVICE", "cpu").strip().lower()
WHISPER_COMPUTE_TYPE = os.getenv("IRIS_WHISPER_COMPUTE_TYPE", "").strip().lower()
WAKE_WORD    = "iris"
WAKE_WORD_RE = re.compile(r"\b(?:hey\s+)?iris\b")
WAKE_TARGETS = (
    "iris",
    "hey iris",
    "hi iris",
)
WAKE_COMMON_MISHEARINGS = {
    "here is",
    "hey i raise",
    "hey i was",
    "hey i miss",
    "yeah i miss",
}

SMART_KEYWORDS = {
    # coding
    "code", "python", "javascript", "bug", "debug", "algorithm", "function",
    "class", "api", "database", "sql", "script", "error", "traceback",
    # reasoning
    "explain", "difference", "compare", "how", "why", "analyze", "pros", "cons",
    "should", "recommend", "best", "way",
    # math
    "calculate", "solve", "equation", "percent", "formula",
}

REALTIME_KEYWORDS = {
    "news", "today", "latest", "current", "price", "score", "won",
    "happened", "right", "now", "recently", "update",
}

FILLER_ONLY_COMMANDS = {
    "please", "pls", "ok", "okay", "hmm", "uh", "um", "yes", "no",
    "thanks", "thank you", "alright", "fine",
}

SLEEP_WORDS  = ["sleep iris", "iris sleep", "standby iris", "go to standby"]
EXIT_WORDS   = ["goodbye iris", "goodbye", "good bye iris", "good bye", "exit iris", "terminate yourself", "shut yourself down"]
STOP_PHRASES = ["stop", "pause", "shut up", "quiet", "mute", "silence"]
SPEECH_STOP_PHRASES = STOP_PHRASES + ["stop iris", "iris stop", "be quiet", "enough", "that is enough", "thats enough"]
PLAY_PHRASES = ["play", "youtube", "song", "music"]
WEATHER_PHRASES = ["weather", "forecast", "temperature", "rain", "snow", "conditions"]
DADDY_HOME_PHRASES = [
    "wake up iris",
    "iris wake up",
    "wake iris up",
]

DADDY_HOME_COMMON_MISHEARINGS = {
    "wake up airis",
    "wake up here is",
    "wake up hey iris",
    "wake up a iris",
    "wake up irish",
}

DADDY_HOME_MUSIC = Path(__file__).resolve().parent / "The Clash - Should I Stay or Should I Go (Official Audio).mp3"

SILENCE_THRESHOLD = 0.04
SILENCE_DURATION  = 1.0
ACTIVE_SILENCE_DURATION = 3.0
ACTIVE_WAIT_TIMEOUT = 5.0
ACTIVE_HARD_MAX_DURATION = 16.0
ACTIVE_MAX_AFTER_SPEECH_DURATION = 9.0
KOKORO_MODEL_FILE = Path(__file__).resolve().parent / "kokoro_models/kokoro-v0_19.onnx"
KOKORO_VOICES_FILE = Path(__file__).resolve().parent / "kokoro_models/voices.json"
KOKORO_VOICE = os.getenv("IRIS_KOKORO_VOICE", "bf_emma")
kokoro = None


def ensure_kokoro_voice_archive():
    archive_file = KOKORO_VOICES_FILE.with_suffix(".npz")
    try:
        if archive_file.exists() and not KOKORO_VOICES_FILE.exists():
            return str(archive_file)

        if archive_file.exists() and KOKORO_VOICES_FILE.exists() and archive_file.stat().st_mtime >= KOKORO_VOICES_FILE.stat().st_mtime:
            return str(archive_file)

        with open(KOKORO_VOICES_FILE, "r", encoding="utf-8") as handle:
            voices_data = json.load(handle)

        if not isinstance(voices_data, dict):
            raise ValueError("voices.json must contain a JSON object")

        normalized_voices = {
            voice_name: np.asarray(values, dtype=np.float32)
            for voice_name, values in voices_data.items()
        }
        np.savez_compressed(archive_file, **normalized_voices)
        return str(archive_file)
    except Exception as e:
        logging.warning("Kokoro voices conversion failed: %s", e)
        return str(KOKORO_VOICES_FILE)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("phonemizer.backend.espeak.words_mismatch").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

current_player = None
is_active      = False
is_speaking    = False
is_processing  = False
stop_speaking_flag = threading.Event()
interrupt_audio_queue = queue.Queue()
app_running    = True
gui            = None


class IrisVisualizer:
    """Simple animated circular waveform UI for Iris states."""

    COLORS = {
        "standby": "#6d8a96",
        "listening": "#2dd4bf",
        "processing": "#ffb347",
        "speaking": "#ff7b54",
    }

    SETTINGS = {
        "standby": (5.0, 0.06, 115.0),
        "listening": (11.0, 0.16, 120.0),
        "processing": (16.0, 0.22, 122.0),
        "speaking": (24.0, 0.30, 126.0),
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Iris")
        self.root.geometry("520x560")
        self.root.configure(bg="#071018")

        self.mode = "standby"
        self.caption = "Say 'Hey Iris'"
        self.phase = 0.0
        self.events = queue.SimpleQueue()
        self.running = True

        self.canvas = tk.Canvas(self.root, bg="#071018", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(30, self._tick)

    def _on_close(self):
        global app_running
        self.running = False
        app_running = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def post_mode(self, mode):
        self.events.put(("mode", mode))

    def post_caption(self, caption):
        self.events.put(("caption", caption))

    def request_close(self):
        self.events.put(("close", None))

    def _apply_events(self):
        while not self.events.empty():
            event, value = self.events.get()
            if event == "mode" and value in self.SETTINGS:
                self.mode = value
            elif event == "caption" and isinstance(value, str):
                self.caption = value[:80]
            elif event == "close":
                self._on_close()

    def _draw_orb(self):
        width = max(self.canvas.winfo_width(), 520)
        height = max(self.canvas.winfo_height(), 560)
        cx, cy = width / 2, height / 2 - 18
        amp, speed, base_radius = self.SETTINGS.get(self.mode, self.SETTINGS["standby"])
        color = self.COLORS.get(self.mode, self.COLORS["standby"])

        self.phase += speed

        points = []
        for deg in range(0, 360, 6):
            ang = math.radians(deg)
            wave = amp * (
                0.60 * math.sin(3.0 * ang + self.phase)
                + 0.40 * math.sin(7.0 * ang - 1.3 * self.phase)
            )
            radius = base_radius + wave
            points.extend([cx + radius * math.cos(ang), cy + radius * math.sin(ang)])

        self.canvas.delete("all")
        self.canvas.create_oval(cx - 98, cy - 98, cx + 98, cy + 98, fill="#0c1b28", outline="")
        self.canvas.create_polygon(points, outline=color, fill="", width=3, smooth=True)
        self.canvas.create_oval(cx - 26, cy - 26, cx + 26, cy + 26, fill=color, outline="")

        self.canvas.create_text(
            cx,
            cy + 145,
            text=f"Iris • {self.mode}",
            fill="#d8ecf2",
            font=("Helvetica", 16, "bold"),
        )
        self.canvas.create_text(
            cx,
            cy + 178,
            text=self.caption,
            fill="#90a9b5",
            font=("Helvetica", 12),
        )

    def _tick(self):
        if not self.running:
            return
        self._apply_events()
        self._draw_orb()
        self.root.after(30, self._tick)

    def run(self):
        self.root.mainloop()


def ui_mode(mode):
    if gui:
        gui.post_mode(mode)


def ui_caption(text):
    if gui:
        gui.post_caption(text)

print("🛠️  Iris Fast starting...")
print(f"Loading Whisper {WHISPER_MODEL_NAME}...")

if not WHISPER_COMPUTE_TYPE:
    # GPU generally performs best with float16; CPU remains int8 by default.
    WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8"

whisper_model = WhisperModel(
    WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE,
    download_root=str(Path.home() / ".cache/whisper")
)

try:
    kokoro = Kokoro(str(KOKORO_MODEL_FILE), ensure_kokoro_voice_archive())
    print("✓ Kokoro TTS ready (bf_emma)")
except Exception as e:
    logging.warning("Kokoro init failed: %s", e)
    kokoro = None

mem.set_model(MEMORY_MODEL)

memory    = mem.load()
mem.trim_conversation_log()
user_name = memory["user"].get("name") or "sir"

print(f"✓ Ready — user: {user_name}, interactions: {memory['interaction_count']}")
print(f"Models → fast: {MODEL_FAST}, smart: {MODEL_SMART}, memory: {MEMORY_MODEL}")
print(f"Whisper → model: {WHISPER_MODEL_NAME}, device: {WHISPER_DEVICE}, compute: {WHISPER_COMPUTE_TYPE}")
print("Say 'Hey Iris' to activate.\n")


def warmup_models():
    """Warm up local models so first request avoids cold-start latency."""
    for model_name in {MODEL_FAST, MODEL_SMART}:
        try:
            ollama.chat(model=model_name, messages=[{"role": "user", "content": "hi"}])
            logging.info("Warmed up model: %s", model_name)
        except Exception as e:
            logging.warning("Warmup failed for %s: %s", model_name, e)


threading.Thread(target=warmup_models, daemon=True).start()


# ──────────────────────────────────────────────
# VOICE
# ──────────────────────────────────────────────

def speak(text, chunked=True):
    global is_speaking
    is_speaking = True
    ui_mode("speaking")
    ui_caption("Speaking...")
    print(f"Iris: {text}")
    try:
        if kokoro:
            chunks = split_spoken_chunks(text) if chunked else [re.sub(r"\s+", " ", (text or "")).strip()]
            for chunk in chunks:
                try:
                    samples, sample_rate = synthesize_kokoro_audio(chunk)
                    sd.play(samples, samplerate=sample_rate)
                    sd.wait()
                except Exception as chunk_error:
                    logging.warning("Kokoro chunk synthesis failed, retrying once: %s", chunk_error)
                    retry_chunk = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", " ", chunk)
                    retry_chunk = re.sub(r"\s+", " ", retry_chunk).strip()
                    if not retry_chunk:
                        continue
                    samples, sample_rate = synthesize_kokoro_audio(retry_chunk)
                    sd.play(samples, samplerate=sample_rate)
                    sd.wait()
        else:
            print("[voice]: Kokoro is not available")

    except Exception as e:
        print(f"[speak error]: {e}")
    finally:
        time.sleep(0.3)
        is_speaking = False
        if is_processing:
            ui_mode("processing")
            ui_caption("Working on it...")
        elif is_active:
            ui_mode("listening")
            ui_caption("Listening")
        else:
            ui_mode("standby")
            ui_caption("Say 'Hey Iris'")


def quick_transcribe(audio):
    segments, _ = whisper_model.transcribe(
        audio,
        beam_size=1,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
        condition_on_previous_text=False,
    )
    return " ".join(s.text for s in segments).strip().lower().strip(".,!? ")


def is_stop_speech_command(text):
    normalized = re.sub(r"[^a-z\s]", " ", (text or "").lower())
    normalized = " ".join(normalized.split())
    if not normalized:
        return False
    if len(normalized.split()) > 3:
        return False
    return any(phrase in normalized for phrase in SPEECH_STOP_PHRASES)


def interrupt_callback(indata, frames, time_info, status):
    """This runs on a separate audio thread automatically."""
    if is_speaking:
        interrupt_audio_queue.put(indata.copy())


def interrupt_listener():
    """Processes audio from the callback queue."""
    buffer = []

    while app_running:
        if not is_speaking:
            while not interrupt_audio_queue.empty():
                try:
                    interrupt_audio_queue.get_nowait()
                except Exception:
                    pass
            buffer = []
            time.sleep(0.05)
            continue

        try:
            chunk = interrupt_audio_queue.get(timeout=0.5)
            chunk = np.squeeze(chunk)
            buffer.append(chunk)

            if len(buffer) < 6:
                continue

            audio = np.concatenate(buffer)
            buffer = []

            volume = np.abs(audio).mean()
            if volume < SILENCE_THRESHOLD:
                continue

            heard = quick_transcribe(audio)
            if heard and is_stop_speech_command(heard):
                print(f"[Interrupt]: '{heard}' — stopping speech")
                stop_speaking_flag.set()
                sd.stop()

        except queue.Empty:
            buffer = []
        except Exception as e:
            logging.warning("Interrupt listener error: %s", e)
            time.sleep(0.2)


def chat_with_retry(messages, model, retries=2, delay=0.8):
    """Retry local LLM calls to smooth transient failures."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return ollama.chat(model=model, messages=messages)
        except Exception as e:
            last_error = e
            if attempt < retries:
                wait = delay * (2 ** attempt)
                logging.warning(
                    "Ollama call failed for %s (attempt %s/%s): %s",
                    model, attempt + 1, retries + 1, e,
                )
                time.sleep(wait)
            else:
                logging.error("Ollama call failed after retries for %s: %s", model, e)
    raise last_error


def stream_and_speak(messages, model):
    """Stream Ollama tokens and speak full sentences as soon as they complete."""
    sentence_buffer = ""
    full_reply = ""

    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            token = ((chunk or {}).get("message") or {}).get("content", "")
            if not token:
                continue

            sentence_buffer += token
            full_reply += token

            # Speak on sentence boundaries for lower perceived latency.
            while True:
                match = re.search(r"(.+?[.!?])(?:\s+|$)", sentence_buffer)
                if not match:
                    break

                sentence = match.group(1).strip()
                sentence_buffer = sentence_buffer[match.end():].lstrip()
                if len(sentence.split()) > 3:
                    speak(sentence, chunked=False)

        leftover = sentence_buffer.strip()
        if leftover and len(leftover.split()) > 2:
            speak(leftover, chunked=False)

        return full_reply.strip()
    except Exception as e:
        logging.warning("Streaming reply failed, using non-stream fallback: %s", e)
        response = chat_with_retry(messages, model=model)
        fallback_reply = response["message"]["content"].strip()
        if fallback_reply:
            speak(fallback_reply)
        return fallback_reply


def compact_spoken_reply(text, max_sentences=2, max_chars=260):
    """Keep replies short for smooth TTS and lower CPU/audio blocking time."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"^(as\s+(an\s+)?ai[^.?!]*[.?!]\s*)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(i\s*(am|'m)\s*phi[^.?!]*[.?!]\s*)", "", cleaned, flags=re.IGNORECASE)

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    clipped = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        clipped.append(sentence)
        if len(clipped) >= max_sentences:
            break

    short = " ".join(clipped) if clipped else cleaned
    if len(short) > max_chars:
        short = short[:max_chars].rsplit(" ", 1)[0].rstrip(".,;: ") + "."
    return short


def split_spoken_chunks(text):
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []

    chunks = []
    for piece in re.split(r"(?<=[.!?])\s+", cleaned):
        piece = piece.strip()
        if piece:
            chunks.append(piece)

    return chunks[:3] if chunks else [cleaned]


def synthesize_kokoro_audio(text):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return kokoro.create(
        text,
        voice=KOKORO_VOICE,
        speed=1.05,
        lang="en-gb",
        trim=False,
    )


def web_search_answer(query, max_results=3):
    """Fetch real-time facts to ground small-model responses."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None
        snippets = " | ".join((r.get("body") or "")[:200] for r in results[:3] if r.get("body"))
        return snippets or None
    except Exception as e:
        logging.warning("Web search failed: %s", e)
        return None


def parse_reminder_command(command):
    """
    Parse natural language reminder commands.
    Returns (reminder_text, remind_datetime) or (None, None).
    """
    now = datetime.now()
    command_lower = command.lower()

    match = re.search(
        r"remind me in (\d+)\s*(minute|minutes|min|hour|hours|hr)\s*(?:to\s+)?(.+)",
        command_lower,
    )
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        text = match.group(3).strip().rstrip(".,?!")
        if "hour" in unit or "hr" in unit:
            delta = amount * 3600
        else:
            delta = amount * 60
        remind_at = datetime.fromtimestamp(now.timestamp() + delta)
        return text, remind_at.strftime("%Y-%m-%d %H:%M")

    match = re.search(
        r"remind me at (\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to\s+)?(.+)",
        command_lower,
    )
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)
        text = match.group(4).strip().rstrip(".,?!")
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        remind_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if remind_at < now:
            remind_at = remind_at.replace(day=remind_at.day + 1)
        return text, remind_at.strftime("%Y-%m-%d %H:%M")

    return None, None


def parse_volume_command(command):
    """Parse volume commands and return a structured action."""
    normalized = normalize_command_text(command)
    if not normalized:
        return None

    if re.search(r"\bmute\b", normalized):
        return {"mode": "mute"}

    absolute_patterns = [
        r"(?:set\s+)?(?:the\s+)?(?:volume|sound)(?:\s+level)?\s*(?:to|at)\s*(\d{1,3})\s*%?",
        r"(?:set\s+)?(?:the\s+)?(?:volume|sound)\s*(\d{1,3})\s*%",
        r"(?:decrease|increase|raise|lower|turn\s+up|turn\s+down)\s+(?:the\s+)?(?:volume|sound)\s*(?:to|at)\s*(\d{1,3})\s*%?",
    ]
    for pattern in absolute_patterns:
        match = re.search(pattern, normalized)
        if match:
            level = max(0, min(100, int(match.group(1))))
            if level == 0:
                return {"mode": "mute"}
            return {"mode": "absolute", "level": level}

    relative_patterns = [
        (r"\b(?:volume|sound)\s*(?:up|higher|increase|raise|louder)\b", "up"),
        (r"\b(?:volume|sound)\s*(?:down|lower|decrease|reduce|quieter)\b", "down"),
        (r"\b(?:turn\s+up|make\s+it\s+louder|increase|raise)\s+(?:the\s+)?(?:volume|sound)\b", "up"),
        (r"\b(?:turn\s+down|make\s+it\s+quieter|decrease|lower|reduce)\s+(?:the\s+)?(?:volume|sound)\b", "down"),
    ]
    for pattern, direction in relative_patterns:
        if re.search(pattern, normalized):
            return {"mode": "relative", "direction": direction, "amount": 10}

    if any(word in normalized for word in ["volume", "sound", "louder", "quieter", "increase", "decrease", "raise", "lower"]):
        return {"mode": "relative", "direction": "up" if any(word in normalized for word in ["up", "higher", "louder", "increase", "raise"]) else "down", "amount": 10}

    return None


def parse_brightness_command(command):
    """Parse brightness commands and return a structured action."""
    normalized = normalize_command_text(command)
    if not normalized:
        return None

    absolute_patterns = [
        r"(?:set\s+)?(?:the\s+)?brightness(?:\s+level)?\s*(?:to|at)\s*(\d{1,3})\s*%?",
        r"(?:set\s+)?(?:the\s+)?screen\s+brightness\s*(?:to|at)\s*(\d{1,3})\s*%?",
    ]
    for pattern in absolute_patterns:
        match = re.search(pattern, normalized)
        if match:
            level = max(0, min(100, int(match.group(1))))
            return {"mode": "absolute", "level": level}

    relative_patterns = [
        (r"\bbrightness\s*(?:up|higher|increase|raise|brighter)\b", "up"),
        (r"\bbrightness\s*(?:down|lower|decrease|reduce|dimmer)\b", "down"),
        (r"\b(?:turn\s+up|make\s+it\s+brighter|increase|raise)\s+(?:the\s+)?brightness\b", "up"),
        (r"\b(?:turn\s+down|make\s+it\s+dimmer|decrease|lower|reduce)\s+(?:the\s+)?brightness\b", "down"),
    ]
    for pattern, direction in relative_patterns:
        if re.search(pattern, normalized):
            return {"mode": "relative", "direction": direction, "amount": 10}

    if any(word in normalized for word in ["brightness", "brighter", "dimmer", "increase", "decrease", "raise", "lower"]):
        return {"mode": "relative", "direction": "up" if any(word in normalized for word in ["up", "higher", "brighter", "increase", "raise"]) else "down", "amount": 10}

    return None


def needs_web_search(command):
    normalized = re.sub(r"[^a-z0-9\s]", " ", command.lower())
    words = set(normalized.split())
    return bool(words.intersection(REALTIME_KEYWORDS))


def select_chat_model(command):
    """Route coding/reasoning-heavy prompts to the smarter model."""
    normalized = command.lower()
    words = set(re.sub(r"[^a-z0-9\s]", " ", normalized).split())

    info_query_prefixes = (
        "tell me about",
        "can you tell me",
        "what is",
        "what are",
        "who is",
        "explain",
    )
    if normalized.startswith(info_query_prefixes):
        return MODEL_SMART

    if words.intersection(SMART_KEYWORDS):
        return MODEL_SMART

    if len(command.split()) > 15:
        return MODEL_SMART

    return MODEL_FAST


def extract_primary_topic(command):
    cleaned = re.sub(r"[^a-z0-9+#\s]", " ", command.lower())
    cleaned = re.sub(
        r"\b(do you know|can you|could you|tell me|about|what is|what's|explain|please|i want to know|give me)\b",
        " ",
        cleaned,
    )
    stop_words = {
        "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are", "me", "you", "it", "this",
    }
    words = [w for w in cleaned.split() if w not in stop_words]
    deduped = []
    for word in words:
        if not deduped or deduped[-1] != word:
            deduped.append(word)
    if not deduped:
        return "this topic"
    return " ".join(deduped[:4])


def should_skip_history_item(text):
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    normalized = " ".join(normalized.split())
    if not normalized:
        return True
    if normalized in FILLER_ONLY_COMMANDS:
        return True

    control_phrases = EXIT_WORDS + SLEEP_WORDS + STOP_PHRASES
    if any(phrase in normalized for phrase in control_phrases):
        return True

    return False


def get_clean_recent_history(memory, n=2):
    history = mem.get_recent_history(memory, n=10)
    filtered = [item for item in history if not should_skip_history_item(item.get("content", ""))]
    return filtered[-n:]


# ──────────────────────────────────────────────
# WEATHER
# ──────────────────────────────────────────────

def normalize_location_text(text):
    cleaned = re.sub(r"\b(weather|forecast|temperature|today|tomorrow|now|current|please|check|tell me|what's|whats|show|give me)\b", " ", text.lower())
    cleaned = re.sub(r"[^a-z0-9,\s.-]", " ", cleaned)
    return " ".join(cleaned.split()).strip(" ,.-")


def extract_weather_location(command):
    patterns = [
        r"\bweather\s+(?:in|for|at)\s+(.+)$",
        r"\bforecast\s+(?:in|for|at)\s+(.+)$",
        r"\btemperature\s+(?:in|for|at)\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, command, re.IGNORECASE)
        if match:
            location = normalize_location_text(match.group(1))
            if location:
                return location

    return None


def fetch_json(url):
    request = Request(url, headers={"User-Agent": "Iris/1.0"})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def weather_location_candidates(location_query):
    base = normalize_location_text(location_query)
    candidates = []

    def add_candidate(value):
        value = normalize_location_text(value)
        if value and value not in candidates:
            candidates.append(value)

    add_candidate(base)

    if "," in base:
        parts = [part.strip() for part in base.split(",") if part.strip()]
        if len(parts) > 1:
            add_candidate(", ".join(parts[-2:]))
            add_candidate(parts[-1])
            add_candidate(parts[0])

    if "district" in base:
        add_candidate(base.replace("district", ""))
    if "province" in base:
        add_candidate(base.replace("province", ""))
    if "zone" in base:
        add_candidate(base.replace("zone", ""))
    if "state" in base:
        add_candidate(base.replace("state", ""))

    add_candidate(base.replace("nepal", "nepal"))
    add_candidate(base.replace("district", "").replace("province", "").replace("zone", ""))

    return candidates


def weather_code_to_text(code):
    descriptions = {
        0: "clear sky",
        1: "mostly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "freezing fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "freezing rain",
        71: "light snow",
        73: "moderate snow",
        75: "heavy snow",
        80: "rain showers",
        81: "strong rain showers",
        82: "violent rain showers",
        95: "thunderstorm",
        96: "thunderstorm with hail",
        99: "thunderstorm with heavy hail",
    }
    return descriptions.get(code, f"weather code {code}")


def get_weather_report(location_query, include_forecast=True):
    if not location_query:
        return None, "I need a location first."

    try:
        results = []
        tried_queries = []

        for candidate in weather_location_candidates(location_query):
            tried_queries.append(candidate)
            geo_url = (
                "https://geocoding-api.open-meteo.com/v1/search?"
                f"name={quote_plus(candidate)}&count=1&language=en&format=json"
            )
            geo_data = fetch_json(geo_url)
            results = geo_data.get("results") or []
            if results:
                break

        if not results:
            return None, f"I couldn't find a place matching {location_query}."

        place = results[0]
        latitude = place["latitude"]
        longitude = place["longitude"]
        display_name = ", ".join(
            part for part in [place.get("name"), place.get("admin1"), place.get("country")]
            if part
        )

        forecast_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}"
            "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            "&timezone=auto"
        )
        weather_data = fetch_json(forecast_url)

        current = weather_data.get("current") or {}
        current_temp = current.get("temperature_2m")
        feels_like = current.get("apparent_temperature")
        wind_speed = current.get("wind_speed_10m")
        humidity = current.get("relative_humidity_2m")
        weather_desc = weather_code_to_text(current.get("weather_code", -1))

        parts = []

        if current_temp is not None:
            current_line = f"Current weather in {display_name}: {current_temp:.0f}°C"
            if feels_like is not None:
                current_line += f", feels like {feels_like:.0f}°C"
            current_line += f", {weather_desc}"
            if wind_speed is not None:
                current_line += f", wind {wind_speed:.0f} km/h"
            if humidity is not None:
                current_line += f", humidity {humidity:.0f}%"
            parts = [current_line + "."]
        else:
            parts = [f"Current weather in {display_name}: {weather_desc}."]

        if include_forecast:
            daily = weather_data.get("daily") or {}
            dates = daily.get("time") or []
            highs = daily.get("temperature_2m_max") or []
            lows = daily.get("temperature_2m_min") or []
            rain = daily.get("precipitation_probability_max") or []
            codes = daily.get("weather_code") or []

            forecast_bits = []
            for index in range(min(2, len(dates))):
                day_name = "Today" if index == 0 else "Tomorrow"
                day_desc = weather_code_to_text(codes[index]) if index < len(codes) else "weather"
                high = highs[index] if index < len(highs) else None
                low = lows[index] if index < len(lows) else None
                rain_chance = rain[index] if index < len(rain) else None

                detail = f"{day_name}: {day_desc}"
                if high is not None and low is not None:
                    detail += f", high {high:.0f}°C low {low:.0f}°C"
                if rain_chance is not None:
                    detail += f", rain chance {rain_chance:.0f}%"
                forecast_bits.append(detail)

            if forecast_bits:
                parts.append("Forecast: " + "; ".join(forecast_bits) + ".")

        return " ".join(parts), None

    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError) as e:
        logging.exception("Weather lookup failed")
        return None, f"I couldn't fetch the weather right now: {str(e)[:80]}"


def handle_weather_request(command):
    location_query = extract_weather_location(command)
    if not location_query:
        location_query = memory.get("user", {}).get("location")

    if not location_query:
        return None, "Which location should I check?"

    include_forecast = not any(phrase in command for phrase in ["current weather", "right now only", "now only"])
    report, error = get_weather_report(location_query, include_forecast=include_forecast)
    if report:
        return report, None
    return None, error


# ──────────────────────────────────────────────
# LISTENING
# ──────────────────────────────────────────────

def record_until_silence(
    max_duration=10,
    silence_duration=SILENCE_DURATION,
    wait_for_speech_timeout=None,
    hard_max_duration=None,
    max_after_speech_duration=None,
):
    fs               = 16000
    chunk_size       = int(fs * 0.1)
    max_chunks       = int(max_duration / 0.1) if max_duration is not None else None
    wait_chunks      = int(wait_for_speech_timeout / 0.1) if wait_for_speech_timeout is not None else None
    hard_max_chunks  = int(hard_max_duration / 0.1) if hard_max_duration is not None else None
    after_speech_chunks = int(max_after_speech_duration / 0.1) if max_after_speech_duration is not None else None
    need_silence     = max(1, int(silence_duration / 0.1))
    audio_chunks     = []
    silence_count    = 0
    started_speaking = False
    chunk_index = 0
    speech_start_chunk = None
    ambient_samples = []
    dynamic_threshold = SILENCE_THRESHOLD

    with sd.InputStream(samplerate=fs, channels=1, dtype=np.float32) as stream:
        while True:
            if max_chunks is not None and chunk_index >= max_chunks:
                break
            if hard_max_chunks is not None and chunk_index >= hard_max_chunks:
                break
            chunk_index += 1

            try:
                chunk, _ = stream.read(chunk_size)
            except Exception as e:
                logging.warning("Audio stream read failed: %s", e)
                break

            chunk    = np.squeeze(chunk)
            volume   = np.abs(chunk).mean()

            if not started_speaking and len(ambient_samples) < 20:
                ambient_samples.append(float(volume))
                ambient_floor = sum(ambient_samples) / len(ambient_samples)
                dynamic_threshold = max(SILENCE_THRESHOLD, ambient_floor * 2.6)

            speech_threshold = dynamic_threshold
            silence_threshold = max(SILENCE_THRESHOLD * 0.7, dynamic_threshold * 0.82)

            if is_speaking:
                continue

            if volume > speech_threshold:
                audio_chunks.append(chunk)
                started_speaking = True
                silence_count    = 0
                if speech_start_chunk is None:
                    speech_start_chunk = chunk_index
            elif started_speaking:
                audio_chunks.append(chunk)
                if volume < silence_threshold:
                    silence_count += 1
                else:
                    silence_count = 0
                if silence_count >= need_silence:
                    break

            if not started_speaking and wait_chunks is not None and chunk_index >= wait_chunks:
                break

            if (
                started_speaking
                and speech_start_chunk is not None
                and after_speech_chunks is not None
                and (chunk_index - speech_start_chunk) >= after_speech_chunks
            ):
                break

    if not audio_chunks:
        return np.zeros(fs, dtype=np.float32), fs
    return np.concatenate(audio_chunks), fs


def transcribe(audio):
    segments, _ = whisper_model.transcribe(
        audio, beam_size=1, language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        initial_prompt=(
            "User may say: hey iris, open firefox, open dolphin, open discord, "
            "open steam, open terminal, open intellij, open vs code"
        ),
        condition_on_previous_text=False,
    )
    text = " ".join(s.text for s in segments).strip().lower().strip(".,!? ")
    if text:
        print(f"[Heard]: {text}")
    return text


def normalize_for_wake(text):
    normalized = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(normalized.split())


def normalize_command_text(text):
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return " ".join(normalized.split())


def _contains_phrase(normalized_text, phrase):
    escaped = re.escape(phrase)
    return bool(re.search(rf"\b{escaped}\b", normalized_text))


def command_has_phrase(command_text, phrase):
    normalized = normalize_command_text(command_text)
    if not normalized:
        return False
    return _contains_phrase(normalized, phrase)


def is_exit_intent(text):
    normalized = normalize_command_text(text)
    if not normalized:
        return False

    exit_phrases = {
        "goodbye",
        "good bye",
        "goodbye iris",
        "good bye iris",
        "bye",
        "bye bye",
        "exit",
        "exit iris",
        "quit",
        "shutdown",
        "shut down",
        "farewell",
        "terminate yourself",
        "shut yourself down",
    }

    for phrase in exit_phrases:
        if normalized == phrase or _contains_phrase(normalized, phrase):
            return True

    return False


def is_standby_intent(text):
    normalized = normalize_command_text(text)
    if not normalized:
        return False

    standby_phrases = {
        "sleep",
        "sleep iris",
        "iris sleep",
        "standby",
        "stand by",
        "standby mode",
        "stand by mode",
        "go to standby",
        "go to standby mode",
        "stay on standby mode",
        "to standby mode",
        "go to sleep",
        "terminal mode",
        "go to terminal mode",
        "time by mode",
    }

    for phrase in standby_phrases:
        if normalized == phrase or _contains_phrase(normalized, phrase):
            return True

    words = normalized.split()
    if not words:
        return False

    for phrase in standby_phrases:
        phrase_words = phrase.split()
        min_len = max(1, len(phrase_words) - 1)
        max_len = min(len(words), len(phrase_words) + 1)
        for n in range(min_len, max_len + 1):
            for i in range(0, len(words) - n + 1):
                candidate = " ".join(words[i:i + n])
                ratio = difflib.SequenceMatcher(None, candidate, phrase).ratio()
                if ratio >= 0.78:
                    return True

    return False


def is_wake_word_detected(text):
    normalized = normalize_for_wake(text)
    if not normalized:
        return False

    if WAKE_WORD_RE.search(normalized):
        return True

    if normalized in WAKE_COMMON_MISHEARINGS:
        return True

    words = normalized.split()
    windows = [" ".join(words[i:i + 2]) for i in range(max(0, len(words) - 1))]
    windows.append(normalized)

    for candidate in windows:
        for target in WAKE_TARGETS:
            ratio = difflib.SequenceMatcher(None, candidate, target).ratio()
            if ratio >= 0.74:
                return True
    return False


def is_daddy_home_detected(text):
    """Detect daddy-home intent with tolerant matching for ASR mishearing."""
    normalized = normalize_for_wake(text)
    if not normalized:
        return False

    def _norm_phrase(phrase):
        return normalize_for_wake(phrase)

    daddy_targets = {_norm_phrase(p) for p in DADDY_HOME_PHRASES}
    daddy_targets.update(DADDY_HOME_COMMON_MISHEARINGS)

    if any(target in normalized for target in daddy_targets):
        return True

    words = normalized.split()
    if not words:
        return False

    windows = [normalized]
    max_n = min(6, len(words))
    for n in range(4, max_n + 1):
        for i in range(0, len(words) - n + 1):
            windows.append(" ".join(words[i:i + n]))

    for candidate in windows:
        for target in daddy_targets:
            ratio = difflib.SequenceMatcher(None, candidate, target).ratio()
            if ratio >= 0.72:
                return True
    return False


# ──────────────────────────────────────────────
# MUSIC
# ──────────────────────────────────────────────

def play_youtube(query):
    global current_player
    search_query = (
        query.replace("play","").replace("song","").replace("on youtube","")
             .replace("youtube","").replace("music","").replace("can you","")
             .replace("please","").strip(" .,?!")
    )
    if not search_query:
        speak("What would you like me to play?")
        return
    speak(f"Playing {search_query}.")
    stop_music(silent=True)
    try:
        current_player = subprocess.Popen(
            ["mpv", "--no-video", "--really-quiet", f"ytdl://ytsearch1:{search_query}"],
            preexec_fn=os.setsid
        )
    except FileNotFoundError:
        speak("mpv is not installed. Run: sudo pacman -S mpv")


def stop_music(silent=False):
    global current_player
    if current_player and current_player.poll() is None:
        try:
            os.killpg(os.getpgid(current_player.pid), signal.SIGTERM)
        except Exception:
            current_player.terminate()
        current_player = None
        if not silent:
            speak("Music stopped.")
    else:
        if not silent:
            speak("Nothing is playing right now.")


# ──────────────────────────────────────────────
# APPLICATION LAUNCHER
# ──────────────────────────────────────────────

APP_ALIASES = {
    # VS Code - common mishearings
    "vs code": {
        "command": ["code"],
        "aliases": ["vscode", "visual studio code", "bs code", "bc code", "v code", "visual code"],
    },
    # Firefox
    "firefox": {
        "command": ["firefox"],
        "aliases": ["firebox", "fire fox", "firefox browser"],
    },
    # Discord
    "discord": {
        "command": ["discord"],
        "aliases": ["discord app", "open discord"],
    },
    # Steam
    "steam": {
        "command": ["steam"],
        "aliases": ["steam app", "open steam"],
    },
    # IntelliJ
    "intellij": {
        "command": ["idea", "intellij-idea-community-edition", "intellij-idea-ultimate-edition"],
        "aliases": ["intelij", "inteli j", "intellij idea", "idea", "jetbrains idea"],
    },
    # Terminal
    "terminal": {
        "command": ["konsole", "gnome-terminal", "xterm"],
        "aliases": ["bash", "shell", "console", "command line", "terminal emulator", "term"],
    },
    # File Manager (CachyOS/KDE default first)
    "file manager": {
        "command": ["dolphin", "nautilus", "thunar"],
        "aliases": ["file explorer", "files", "file browser", "dolphin", "the door thing", "door thing"],
    },
}


def clean_app_request(app_name):
    cleaned = re.sub(
        r"\b(open|launch|start|can you|could you|please|me|i said|just|now|for me|run|the|an|a|app|application|instance|that is already running|already running)\b",
        " ",
        (app_name or ""),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,?!")
    return cleaned


def resolve_launch_command(command_entry):
    commands = command_entry if isinstance(command_entry, list) else [command_entry]
    for cmd in commands:
        if cmd and shutil.which(cmd):
            return cmd
    return None


def find_app_command(app_name, memory=None):
    """
    Find and correct mishearings in app names.
    Returns (app_display_name, command) or (None, None) if not found.
    """
    normalized = re.sub(r"[^a-z0-9\s]", " ", clean_app_request(app_name).lower()).strip()
    normalized = " ".join(normalized.split())

    if not normalized:
        return None, None

    # Check user-learned corrections first.
    if memory:
        app_corrections = memory.get("app_corrections", {})
        for learned_name, display in app_corrections.items():
            if normalized == learned_name or difflib.SequenceMatcher(
                None, normalized, learned_name
            ).ratio() >= 0.80:
                # Look up the corrected name in static aliases.
                result = _find_in_aliases(display.lower())
                if result[0]:
                    return result

    return _find_in_aliases(normalized)


def _find_in_aliases(normalized):
    """Original alias lookup logic extracted into helper."""
    for app_display, app_info in APP_ALIASES.items():
        if normalized == app_display or normalized == app_display.replace(" ", ""):
            return app_display, app_info["command"]

    # Check aliases with fuzzy matching
    for app_display, app_info in APP_ALIASES.items():
        for alias in app_info["aliases"]:
            alias_norm = re.sub(r"[^a-z0-9\s]", " ", alias.lower()).strip()
            alias_norm = " ".join(alias_norm.split())

            # Exact alias match
            if normalized == alias_norm or normalized == alias_norm.replace(" ", ""):
                return app_display, app_info["command"]

            # Fuzzy match for common mishearings
            ratio = difflib.SequenceMatcher(None, normalized, alias_norm).ratio()
            if ratio >= 0.75:  # High similarity threshold for mishearings
                return app_display, app_info["command"]

    # If no match found, try levenshtein-like matching on app names
    for app_display, app_info in APP_ALIASES.items():
        ratio = difflib.SequenceMatcher(None, normalized, app_display).ratio()
        if ratio >= 0.70:
            return app_display, app_info["command"]

    return None, None


def launch_app(app_name, memory=None):
    """
    Launch an application with intelligent mishearing correction.
    Provides helpful feedback to the user.
    """
    app_display, command_entry = find_app_command(app_name, memory=memory)
    
    if not app_display:
        speak(f"I couldn't find an app called {app_name}. Try: Discord, File Manager, Steam, Firefox, Terminal, VS Code, or IntelliJ.")
        return

    command = resolve_launch_command(command_entry)
    if not command:
        speak(f"{app_display} doesn't seem to be installed on your system.")
        return
    
    if app_name.lower() != app_display.lower():
        speak(f"I think you meant {app_display}. Opening it now.")
    else:
        speak(f"Opening {app_display}.")
    
    try:
        # Launch the app in background
        subprocess.Popen(
            [command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None
        )
    except FileNotFoundError:
        speak(f"{app_display} is not installed. Please install it and try again.")
    except Exception as e:
        speak(f"Failed to open {app_display}: {str(e)[:60]}")


def close_app(app_name, memory=None):
    app_display, command_entry = find_app_command(app_name, memory=memory)

    if not app_display:
        speak(f"I couldn't find an app called {app_name} to close.")
        return

    if app_display == "terminal":
        speak("I won't close terminal automatically to avoid disrupting your session.")
        return

    commands = command_entry if isinstance(command_entry, list) else [command_entry]
    process_hints = {app_display.lower().replace(" ", "")}
    for cmd in commands:
        if cmd:
            process_hints.add(cmd.lower())
            process_hints.add(Path(cmd).name.lower())

    matched_processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if any(hint and (hint in name or hint in cmdline) for hint in process_hints):
                matched_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if not matched_processes:
        speak(f"{app_display} is not running right now.")
        return

    closed_count = 0
    for proc in matched_processes:
        try:
            proc.terminate()
            closed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if closed_count > 0:
        speak(f"Closed {app_display}.")
    else:
        speak(f"I couldn't close {app_display} due to system permissions.")


def should_treat_as_app_close(command_text):
    normalized = re.sub(r"[^a-z0-9\s]", " ", (command_text or "").lower()).strip()
    normalized = " ".join(normalized.split())
    if not normalized:
        return False

    if not any(word in normalized for word in ["close", "quit", "terminate", "end"]):
        return False

    cleaned = re.sub(r"\b(close|quit|terminate|end|please|can you|could you|for me|the|app|application)\b", " ", normalized)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return False

    app_display, _ = find_app_command(cleaned)
    return bool(app_display)


def is_music_playing():
    return bool(current_player and current_player.poll() is None)


def should_treat_as_app_launch(command_text):
    normalized = re.sub(r"[^a-z0-9\s]", " ", (command_text or "").lower()).strip()
    normalized = " ".join(normalized.split())
    if not normalized:
        return False

    if any(word in normalized for word in ["open", "launch", "start", "run"]):
        return True

    app_display, _ = find_app_command(normalized)
    if app_display and len(normalized.split()) <= 5:
        return True
    return False


# ──────────────────────────────────────────────
# SYSTEM INFO
# ──────────────────────────────────────────────

def get_system_info():
    battery = psutil.sensors_battery()
    battery_str = (
        f"{battery.percent:.0f}% {'charging' if battery.power_plugged else 'on battery'}"
        if battery else "unknown"
    )
    speak(
        f"Time is {datetime.now().strftime('%I:%M %p')}. "
        f"Battery at {battery_str}. "
        f"CPU at {psutil.cpu_percent()}%."
    )


def reminder_checker():
    """Background thread that fires reminders regardless of Iris state."""
    while app_running:
        try:
            pending = mem.get_pending_reminders(memory)
            if pending:
                mem.mark_reminders_fired(memory, pending)
                for reminder in pending:
                    text = reminder.get("text", "something")
                    print(f"[Reminder]: firing — {text}")

                    waited = 0
                    while (is_speaking or is_processing) and waited < 30:
                        time.sleep(1)
                        waited += 1

                    speak(f"Reminder: {text}.")

                    if not is_active:
                        ui_mode("standby")
                        ui_caption("Say 'Hey Iris'")
        except Exception as e:
            logging.warning("Reminder checker error: %s", e)

        has_reminders = any(not r.get("fired") for r in memory.get("reminders", []))
        time.sleep(30 if has_reminders else 120)


def proactive_monitor():
    """
    Fires proactive updates when Iris is active.
    Checks battery, time-based greetings, etc.
    """
    last_morning_check = None
    last_battery_warning = None

    while app_running:
        try:
            if not is_active or is_speaking or is_processing:
                time.sleep(15)
                continue

            now = datetime.now()

            if now.hour in (7, 8, 9) and last_morning_check != now.date():
                last_morning_check = now.date()
                battery = psutil.sensors_battery()
                name = memory["user"].get("name") or "sir"
                if battery:
                    battery_msg = (
                        f"Good morning, {name}. Battery is at {battery.percent:.0f} percent"
                        + (" and charging." if battery.power_plugged else ", not plugged in.")
                    )
                else:
                    battery_msg = f"Good morning, {name}."
                time.sleep(2)
                speak(battery_msg)

            battery = psutil.sensors_battery()
            if (
                battery
                and not battery.power_plugged
                and battery.percent < 20
                and last_battery_warning != now.date()
            ):
                last_battery_warning = now.date()
                time.sleep(1)
                speak(f"Heads up, battery is at {battery.percent:.0f} percent. You should plug in soon.")

        except Exception as e:
            logging.warning("Proactive monitor error: %s", e)

        time.sleep(60)


def set_volume(direction, amount=10):
    """Raise or lower system volume using pactl."""
    try:
        sign = "+" if direction == "up" else "-"
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{amount}%"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        speak(f"Volume {'up' if direction == 'up' else 'down' }.")
    except Exception as e:
        speak("I couldn't adjust the volume.")
        logging.warning("Volume control failed: %s", e)


def set_volume_absolute(percent):
    """Set system volume to an exact percentage using pactl."""
    try:
        level = max(0, min(100, int(percent)))
        if level == 0:
            subprocess.run(
                ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            speak("Muted.")
            return

        subprocess.run(
            ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        speak(f"Volume set to {level} percent.")
    except Exception as e:
        speak("I couldn't set the volume.")
        logging.warning("Absolute volume control failed: %s", e)


def set_brightness(direction, amount=10):
    """Raise or lower screen brightness using brightnessctl."""
    try:
        if not shutil.which("brightnessctl"):
            speak("brightnessctl is not installed. Run: yay -S brightnessctl")
            return
        sign = "+" if direction == "up" else "-"
        subprocess.run(
            ["brightnessctl", "set", f"{amount}%{sign}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        speak(f"Brightness {'up' if direction == 'up' else 'down' }.")
    except Exception as e:
        speak("I couldn't adjust the brightness.")
        logging.warning("Brightness control failed: %s", e)


def set_brightness_absolute(percent):
    """Set screen brightness to an exact percentage using brightnessctl."""
    try:
        if not shutil.which("brightnessctl"):
            speak("brightnessctl is not installed. Run: yay -S brightnessctl")
            return
        level = max(0, min(100, int(percent)))
        subprocess.run(
            ["brightnessctl", "set", f"{level}%"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        speak(f"Brightness set to {level} percent.")
    except Exception as e:
        speak("I couldn't set the brightness.")
        logging.warning("Absolute brightness control failed: %s", e)


def daddy_home_entry():
    """Cinematic Jarvis-style entry sequence."""
    global is_active

    is_active = True
    ui_mode("listening")

    try:
        music_path = str(DADDY_HOME_MUSIC)
        if not DADDY_HOME_MUSIC.exists():
            logging.warning("Daddy home music not found: %s", music_path)
        else:
            global current_player
            stop_music(silent=True)
            current_player = subprocess.Popen(
                ["mpv", "--no-video", "--really-quiet", music_path],
                preexec_fn=os.setsid
            )
    except FileNotFoundError:
        speak("mpv is not installed.")
        return
    except Exception as e:
        logging.warning("Music failed: %s", e)

    time.sleep(1.8)

    name = memory["user"].get("name") or "sir"
    speak(f"Welcome back, {name}. All systems online.")

    time.sleep(0.6)
    _open_window_right("code")
    time.sleep(1.2)
    _open_window_left("firefox")
    time.sleep(0.8)
    speak("What are we working on today?")


def _open_window_right(command):
    """Launch app and snap it to the right half using KWin scripting."""
    _launch_and_snap(command, side="right")


def _open_window_left(command):
    """Launch app and snap it to the left half using KWin scripting."""
    _launch_and_snap(command, side="left")


def _launch_and_snap(command, side):
    """Launch an app and snap it with KWin scripting."""
    try:
        if not shutil.which(command):
            logging.warning("Command not found: %s", command)
            return

        subprocess.Popen(
            [command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )

        time.sleep(2.5)

        if _snap_with_kwin_script(command, side=side):
            return

        logging.warning("Window %s snap failed for command: %s", side, command)
    except Exception as e:
        logging.warning("Window %s launch/snap failed: %s", side, e)


def _snap_with_kwin_script(command, side):
    """Wayland-native KDE snapping via KWin scripting and DBus."""
    qdbus_bin = shutil.which("qdbus6") or shutil.which("qdbus-qt6") or shutil.which("qdbus")
    if not qdbus_bin:
        logging.warning("qdbus not found; cannot use KWin fallback.")
        return False

    target_class = re.sub(r"[^a-z0-9_-]", "", command.lower()) or command.lower()
    side = "right" if side == "right" else "left"
    script_path = None

    try:
        kwin_script = f"""
var targetClass = "{target_class}";
var side = "{side}";

function snapWindow() {{
    var windows = workspace.windowList();
    var target = null;

    for (var i = windows.length - 1; i >= 0; --i) {{
        var w = windows[i];
        if (!w || !w.normalWindow) {{
            continue;
        }}
        var rc = (w.resourceClass || "").toLowerCase();
        var rn = (w.resourceName || "").toLowerCase();
        if (rc.indexOf(targetClass) !== -1 || rn.indexOf(targetClass) !== -1) {{
            target = w;
            break;
        }}
    }}

    if (!target) {{
        return;
    }}

    var area = workspace.clientArea(KWin.MaximizeArea, target);
    var half = Math.floor(area.width / 2);
    var x = (side === "right") ? area.x + half : area.x;

    target.frameGeometry = {{
        x: x,
        y: area.y,
        width: half,
        height: area.height
    }};
}}

snapWindow();
""".strip()

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
            tmp.write(kwin_script)
            script_path = tmp.name

        load = subprocess.run(
            [
                qdbus_bin,
                "org.kde.KWin",
                "/Scripting",
                "org.kde.kwin.Scripting.loadScript",
                script_path,
                f"iris_snap_{target_class}_{int(time.time() * 1000)}",
            ],
            timeout=6,
            check=True,
            text=True,
            capture_output=True,
        )

        script_id_match = re.search(r"\d+", load.stdout or "")
        if not script_id_match:
            logging.warning("KWin script load returned no script id for %s.", command)
            return False

        script_id = script_id_match.group(0)
        subprocess.run(
            [
                qdbus_bin,
                "org.kde.KWin",
                f"/Scripting/Script{script_id}",
                "org.kde.kwin.Script.run",
            ],
            timeout=6,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        logging.warning("KWin %s snap failed for %s: %s", side, command, e)
        return False
    finally:
        if script_path:
            try:
                os.unlink(script_path)
            except Exception:
                pass


# ──────────────────────────────────────────────
# SHUTDOWN
# ──────────────────────────────────────────────

def shutdown():
    global app_running
    app_running = False
    stop_music(silent=True)
    speak("Goodbye sir.")
    if gui:
        gui.request_close()
    os._exit(0)


# ──────────────────────────────────────────────
# COMMANDS
# ──────────────────────────────────────────────

def execute_command(command):
    global memory, user_name, is_active, is_processing

    is_processing = True
    ui_mode("processing")
    ui_caption("Working on it...")
    threading.Thread(target=mem.extract_and_save, args=(memory, command), daemon=True).start()
    mem.add_to_history(memory, "user", command)
    user_name = memory["user"].get("name") or "sir"
    normalized_command = normalize_command_text(command)

    try:
        if normalized_command in FILLER_ONLY_COMMANDS:
            speak("I heard you. Ask me a full question and I will answer properly.")
            return

        if is_music_playing():
            should_process_during_music = (
                is_exit_intent(normalized_command)
                or is_standby_intent(normalized_command)
                or any(command_has_phrase(normalized_command, p) for p in STOP_PHRASES)
                or should_treat_as_app_close(command)
            )
            if not should_process_during_music:
                logging.info("Ignoring likely music bleed command: %s", normalized_command)
                return

        # ── exit completely ──
        if is_exit_intent(normalized_command):
            shutdown()

        # ── sleep / standby — just stop listening actively ──
        elif is_standby_intent(normalized_command) or any(command_has_phrase(normalized_command, p) for p in SLEEP_WORDS) or normalized_command in ["good night"]:
            is_active = False
            name = memory["user"].get("name") or ""
            speak(f"Going to standby. Say 'Hey Iris' when you need me{', ' + name if name else ''}.")

        # ── stop music ──
        elif any(command_has_phrase(normalized_command, p) for p in STOP_PHRASES):
            stop_music()

        # ── close applications ──
        elif should_treat_as_app_close(command):
            cleaned = re.sub(
                r"\b(close|quit|terminate|end|can you|could you|please|for me|the|app|application|i said|just|now)\b",
                "",
                command,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned:
                close_app(cleaned, memory=memory)
            else:
                speak("Which app should I close?")

        # ── play music ──
        elif any(command_has_phrase(normalized_command, p) for p in PLAY_PHRASES):
            play_youtube(command)

        # ── system ──
        elif "battery" in command or "cpu" in command or "status" in command:
            get_system_info()

        elif "what time" in command or ("time" in command and "what" in command):
            speak(f"It is {datetime.now().strftime('%I:%M %p')}.")

        elif any(command_has_phrase(normalized_command, p) for p in WEATHER_PHRASES):
            weather_reply, weather_error = handle_weather_request(command)
            if weather_reply:
                speak(weather_reply)
            else:
                speak(weather_error or "I couldn't check the weather right now.")

        elif "remind me" in normalized_command:
            reminder_text, remind_at = parse_reminder_command(command)
            if reminder_text and remind_at:
                mem.save_reminder(memory, reminder_text, remind_at)
                remind_dt = datetime.strptime(remind_at, "%Y-%m-%d %H:%M")
                time_str = remind_dt.strftime("%I:%M %p")
                speak(f"Got it. I'll remind you to {reminder_text} at {time_str}.")
            else:
                speak("When should I remind you, and what about?")

        elif (volume_request := parse_volume_command(command)):
            if volume_request["mode"] == "mute":
                set_volume_absolute(0)
            elif volume_request["mode"] == "absolute":
                set_volume_absolute(volume_request["level"])
            elif volume_request["direction"] == "up":
                set_volume("up", volume_request.get("amount", 10))
            else:
                set_volume("down", volume_request.get("amount", 10))

        elif (brightness_request := parse_brightness_command(command)):
            if brightness_request["mode"] == "absolute":
                set_brightness_absolute(brightness_request["level"])
            elif brightness_request["direction"] == "up":
                set_brightness("up", brightness_request.get("amount", 10))
            else:
                set_brightness("down", brightness_request.get("amount", 10))

        elif "open brow*ser" in command or "open google" in command:
            speak("Opening browser.")
            webbrowser.open("https://google.com")

        # ── open applications ──
        elif any(p in command for p in ["open", "launch", "start"]):
            cleaned = re.sub(
                r"\b(open|launch|start|can you|could you|please|me|i said|just|now)\b",
                "",
                command,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned:
                launch_app(cleaned, memory=memory)
            else:
                speak("Which app would you like to open?")

        elif should_treat_as_app_launch(command):
            cleaned = clean_app_request(command)
            if cleaned:
                launch_app(cleaned, memory=memory)
            else:
                speak("Which app would you like to open?")

        elif any(p in command for p in ["introduce yourself", "who are you", "tell me about yourself"]):
            speak(
                f"I am Iris, your personal assistant, {user_name}. "
                "I can control music, system actions, web tasks, and remember important details about you."
            )

        # ── memory ──
        elif any(p in command for p in ["where do i live", "where i live", "my location", "where am i from"]):
            known_location = memory.get("user", {}).get("location")
            if known_location:
                speak(f"You live in {known_location}.")
            else:
                speak("I do not have your location saved yet. You can tell me: I live in your city and country.")

        elif any(p in command for p in ["what do you know", "what do you remember", "what you know about me"]):
            speak(mem.get_memory_summary(memory))

        elif "forget everything" in command or "clear memory" in command or "reset memory" in command:
            mem.clear_memory(memory)
            memory = mem.load()
            speak("Memory wiped. Starting fresh.")

        elif "forget that" in command:
            if memory["facts"]:
                removed = memory["facts"].pop()
                mem.save(memory)
                speak(f"Forgotten: {removed}.")
            else:
                speak("Nothing recent to forget.")

        elif any(
            p in normalized_command
            for p in [
                "you said",
                "you meant",
                "its actually",
                "it s actually",
                "it's actually",
                "not",
                "i said",
                "i meant",
                "wrong",
                "mistake",
                "correct",
            ]
        ):
            # Try to extract correction patterns like "X not Y" or "it's X not Y".
            correction_match = re.search(
                r"\bit['\s]*s\s+(.+?)(?:\s+not\s+.+)?$|\bnot\s+\S+\s+(?:its?|it['\s]*s)\s+(.+)$|(?:i said|i meant)\s+(.+)$",
                normalized_command,
            )
            if correction_match:
                corrected = next((g for g in correction_match.groups() if g), None)
                corrected = corrected.strip() if corrected else ""
                if corrected:
                    mem.save_correction(memory, corrected)
                    speak(f"Got it. I'll remember that you meant {corrected}.")
                else:
                    mem.add_to_history(memory, "assistant", "")
                    speak("Sorry about that. Could you repeat what you meant?")
            else:
                # Fall through behavior for conversational clarification.
                mem.add_to_history(memory, "assistant", "")
                speak("Sorry about that. Could you repeat what you meant?")

        # ── general AI ──
        else:
            messages = [{"role": "system", "content": mem.build_system_prompt(memory)}]

            messages.append({
                "role": "system",
                "content": (
                    "Stay focused on the user's latest request. "
                    "Do not add unrelated claims. "
                    "Answer directly first, then add short helpful context."
                ),
            })

            rag_context = mem.build_rag_context(command)
            if rag_context:
                messages.append({"role": "system", "content": rag_context})

            if needs_web_search(command):
                search_data = web_search_answer(command)
                if search_data:
                    messages.append({
                        "role": "system",
                        "content": (
                            "Real-time web data for this query: "
                            f"{search_data}\nUse this data when relevant."
                        ),
                    })

            recent_history = get_clean_recent_history(memory, n=6)
            messages += recent_history
            messages.append({
                "role": "user",
                "content": command,
            })

            selected_model = select_chat_model(command)
            logging.info("Model selected: %s", selected_model)
            reply = stream_and_speak(messages, selected_model)
            reply = compact_spoken_reply(reply)
            if reply and reply[-1] not in ".!?":
                reply += "."

            mem.extract_extended_memory(memory, command, reply)
            mem.log_conversation(command, reply)
            mem.add_to_history(memory, "assistant", reply)

    except SystemExit:
        raise
    except Exception as e:
        speak(f"Minor glitch: {str(e)[:60]}")
    finally:
        is_processing = False
        if is_active and not is_speaking:
            ui_mode("listening")
            ui_caption("Listening")


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def run_assistant_loop():
    global is_active

    try:
        while app_running:
            # Avoid opening mic capture while playback is active; this reduces ALSA contention and lag.
            if is_speaking:
                time.sleep(0.05)
                continue

            if not is_active:
                print("👂 Standby...")
                ui_mode("standby")
                ui_caption("Say 'Hey Iris'")
                # True standby: wait for real speech before returning.
                audio, _ = record_until_silence(max_duration=None)
                text = transcribe(audio)

                if not text:
                    continue

                if is_exit_intent(text):
                    shutdown()

                if is_daddy_home_detected(text):
                    threading.Thread(target=daddy_home_entry, daemon=True).start()

                elif is_wake_word_detected(text):
                    is_active = True
                    ui_mode("listening")
                    ui_caption("Listening")
                    name = memory["user"].get("name")
                    if memory["interaction_count"] == 0:
                        greeting = "Iris online. What's your name?"
                    elif name:
                        greeting = f"Online. Good to have you back, {name}."
                    else:
                        greeting = "Online. What do you need?"
                        speak(greeting, chunked=False)

            else:
                print("👂 Listening...")
                ui_mode("listening")
                ui_caption("Listening")
                # In active mode, wait until user starts speaking, then stop after a longer pause.
                audio, _ = record_until_silence(
                    max_duration=None,
                    silence_duration=ACTIVE_SILENCE_DURATION,
                    wait_for_speech_timeout=ACTIVE_WAIT_TIMEOUT,
                    hard_max_duration=ACTIVE_HARD_MAX_DURATION,
                    max_after_speech_duration=ACTIVE_MAX_AFTER_SPEECH_DURATION,
                )

                # No speech detected: skip Whisper pass to avoid unnecessary CPU load.
                if audio.size == 0 or float(np.max(np.abs(audio))) < (SILENCE_THRESHOLD * 0.8):
                    continue

                text = transcribe(audio)

                if not text:
                    continue

                command = WAKE_WORD_RE.sub("", text).strip(" .,?!")
                if command:
                    print(f"[Command]: {command}")
                    ui_caption(command)
                    # Process one command at a time to avoid delayed/out-of-order replies.
                    execute_command(command)

    except KeyboardInterrupt:
        shutdown()


def main():
    global gui

    gui_env = os.getenv("IRIS_GUI", "1").lower()
    gui_enabled = gui_env not in {"0", "false", "no"}
    has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    tk_available = tk is not None

    use_gui = (
        tk_available
        and gui_enabled
        and has_display
    )

    if use_gui:
        gui = IrisVisualizer()
        worker = threading.Thread(target=run_assistant_loop, daemon=True)
        worker.start()
        threading.Thread(target=reminder_checker, daemon=True).start()
        threading.Thread(target=proactive_monitor, daemon=True).start()
        gui.run()
    else:
        if not gui_enabled:
            print("[GUI] Disabled by IRIS_GUI setting.")
        elif not tk_available:
            print("[GUI] Tkinter is not available in this Python environment.")
        elif not has_display:
            print("[GUI] No DISPLAY/WAYLAND_DISPLAY detected, running headless mode.")
        threading.Thread(target=reminder_checker, daemon=True).start()
        threading.Thread(target=proactive_monitor, daemon=True).start()
        run_assistant_loop()


if __name__ == "__main__":
    main()