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
import iris_skills

# ── optional audio-enhancement libraries ──────────────────────────────────
try:
    import noisereduce as _nr
    _NOISEREDUCE_AVAILABLE = True
except ImportError:
    _NOISEREDUCE_AVAILABLE = False
    logging.warning("noisereduce not installed — skipping noise reduction. "
                    "Run: pip install noisereduce")

try:
    from openwakeword.model import Model as _OWWModel
    _OWW_AVAILABLE = True
except ImportError:
    _OWWModel = None
    _OWW_AVAILABLE = False
    logging.warning("openwakeword not installed — falling back to Whisper wake detection. "
                    "Run: pip install openwakeword")

# ============== IRIS FAST ==============
MODEL_FAST   = os.getenv("IRIS_MODEL_FAST", "phi3.5")
MODEL_SMART  = os.getenv("IRIS_MODEL_SMART", "llama3.1:8b")
MEMORY_MODEL = os.getenv("IRIS_MEMORY_MODEL", "phi3.5")
WHISPER_MODEL_NAME = os.getenv("IRIS_WHISPER_MODEL", "tiny.en")
WHISPER_DEVICE = os.getenv("IRIS_WHISPER_DEVICE", "cpu").strip().lower()
WHISPER_COMPUTE_TYPE = os.getenv("IRIS_WHISPER_COMPUTE_TYPE", "").strip().lower()
WAKE_WORD    = "iris"
WAKE_WORD_RE = re.compile(r"\b(?:hey\s+)?iris\b")
WAKE_TARGETS = ("iris", "hey iris", "hi iris")
WAKE_COMMON_MISHEARINGS = {
    "here is", "hey i raise", "hey i was", "hey i miss", "yeah i miss",
}

SMART_KEYWORDS = {
    "code", "python", "javascript", "bug", "debug", "algorithm", "function",
    "class", "api", "database", "sql", "script", "error", "traceback",
    "explain", "difference", "compare", "how", "why", "analyze", "pros", "cons",
    "should", "recommend", "best", "way",
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
EXIT_WORDS   = ["goodbye iris", "goodbye", "good bye iris", "good bye", "exit iris",
                "terminate yourself", "shut yourself down"]
STOP_PHRASES = ["stop", "pause", "shut up", "quiet", "mute", "silence"]
SPEECH_STOP_PHRASES = STOP_PHRASES + [
    "stop iris", "iris stop", "be quiet", "enough", "that is enough", "thats enough",
]
DADDY_HOME_PHRASES = ["wake up iris", "iris wake up", "wake iris up"]
DADDY_HOME_COMMON_MISHEARINGS = {
    "wake up airis", "wake up here is", "wake up hey iris",
    "wake up a iris", "wake up irish",
}
DADDY_HOME_MUSIC = Path(__file__).resolve().parent / "The Clash - Should I Stay or Should I Go (Official Audio).mp3"

SILENCE_THRESHOLD = 0.04
SILENCE_DURATION  = 1.0

# ── openWakeWord config ────────────────────────────────────────────────────
OWW_WAKE_THRESHOLD     = float(os.getenv("IRIS_OWW_THRESHOLD",       "0.35"))
OWW_DADDY_THRESHOLD    = float(os.getenv("IRIS_OWW_DADDY_THRESHOLD", "0.30"))
OWW_CHUNK_SAMPLES      = 1280
OWW_SAMPLE_RATE        = 16000
OWW_POST_WAKE_RECORD_S = float(os.getenv("IRIS_OWW_POST_WAKE_S", "0.4"))

# ── noise reduction config ─────────────────────────────────────────────────
NR_PROP_DECREASE = float(os.getenv("IRIS_NR_PROP_DECREASE", "0.75"))

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
        if archive_file.exists() and KOKORO_VOICES_FILE.exists() \
                and archive_file.stat().st_mtime >= KOKORO_VOICES_FILE.stat().st_mtime:
            return str(archive_file)
        with open(KOKORO_VOICES_FILE, "r", encoding="utf-8") as handle:
            voices_data = json.load(handle)
        if not isinstance(voices_data, dict):
            raise ValueError("voices.json must contain a JSON object")
        normalized_voices = {
            name: np.asarray(vals, dtype=np.float32)
            for name, vals in voices_data.items()
        }
        np.savez_compressed(archive_file, **normalized_voices)
        return str(archive_file)
    except Exception as e:
        logging.warning("Kokoro voices conversion failed: %s", e)
        return str(KOKORO_VOICES_FILE)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("phonemizer.backend.espeak.words_mismatch").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

current_player = None
is_active      = False
is_speaking    = False
is_processing  = False
stop_speaking_flag   = threading.Event()
interrupt_audio_queue = queue.Queue()
app_running    = True
gui            = None


# ──────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────

class IrisVisualizer:
    """
    Iris HUD — Jarvis-style GUI with:
      • Animated waveform orb (existing)
      • Scrolling conversation transcript (last 5 exchanges)
      • Live stats bar: CPU · RAM · Battery · Clock
      • Mode label with colour coding
    """

    # ── colour palette ─────────────────────────────────────────────────────
    BG          = "#071018"
    PANEL_BG    = "#0b1a26"
    BORDER      = "#112233"
    TEXT_DIM    = "#4a6a7a"
    TEXT_MID    = "#7aaabb"
    TEXT_BRIGHT = "#d8ecf2"
    ACCENT      = {
        "standby":    "#6d8a96",
        "listening":  "#2dd4bf",
        "processing": "#ffb347",
        "speaking":   "#ff7b54",
    }

    ORB_SETTINGS = {
        "standby":    (5.0,  0.06, 72.0),
        "listening":  (11.0, 0.16, 76.0),
        "processing": (16.0, 0.22, 78.0),
        "speaking":   (24.0, 0.30, 82.0),
    }

    # ── layout constants (px) ──────────────────────────────────────────────
    WIN_W         = 580
    WIN_H         = 720
    ORB_AREA_H    = 310
    STATS_H       = 28
    TRANSCRIPT_H  = 330
    PAD           = 14

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Iris")
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self.mode    = "standby"
        self.caption = "Say 'Hey Iris'"
        self.phase   = 0.0
        self.events  = queue.SimpleQueue()
        self.running = True

        # Transcript: list of (role, text) tuples, newest at end
        self._transcript: list[tuple[str, str]] = []
        self._transcript_lock = threading.Lock()

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(30,   self._tick)
        self.root.after(1000, self._tick_stats)

    # ── layout ─────────────────────────────────────────────────────────────

    def _build_layout(self):
        W, H = self.WIN_W, self.WIN_H

        # Orb canvas (top)
        self.orb_canvas = tk.Canvas(
            self.root, width=W, height=self.ORB_AREA_H,
            bg=self.BG, highlightthickness=0,
        )
        self.orb_canvas.place(x=0, y=0)

        # Stats bar
        self.stats_frame = tk.Frame(
            self.root, bg=self.PANEL_BG,
            height=self.STATS_H, width=W,
        )
        self.stats_frame.place(x=0, y=self.ORB_AREA_H)

        self.stats_label = tk.Label(
            self.stats_frame, bg=self.PANEL_BG,
            fg=self.TEXT_DIM, font=("Courier", 9),
            text="", anchor="w", padx=self.PAD,
        )
        self.stats_label.pack(fill="both", expand=True)

        # Separator line
        sep = tk.Frame(self.root, bg=self.BORDER, height=1, width=W)
        sep.place(x=0, y=self.ORB_AREA_H + self.STATS_H)

        # Transcript panel (bottom)
        trans_y = self.ORB_AREA_H + self.STATS_H + 1
        self.trans_canvas = tk.Canvas(
            self.root, width=W, height=self.TRANSCRIPT_H,
            bg=self.PANEL_BG, highlightthickness=0,
        )
        self.trans_canvas.place(x=0, y=trans_y)

    # ── event API (called from worker threads) ─────────────────────────────

    def post_mode(self, mode: str):
        self.events.put(("mode", mode))

    def post_caption(self, caption: str):
        self.events.put(("caption", caption))

    def post_transcript(self, role: str, text: str):
        """Add a line to the conversation transcript."""
        self.events.put(("transcript", (role, text)))

    def request_close(self):
        self.events.put(("close", None))

    # ── internal event pump ────────────────────────────────────────────────

    def _apply_events(self):
        while not self.events.empty():
            try:
                event, value = self.events.get_nowait()
            except Exception:
                break
            if event == "mode" and value in self.ORB_SETTINGS:
                self.mode = value
            elif event == "caption" and isinstance(value, str):
                self.caption = value[:90]
            elif event == "transcript" and isinstance(value, tuple):
                role, text = value
                with self._transcript_lock:
                    self._transcript.append((role, text))
                    # Keep only last 10 lines (5 exchanges × 2)
                    if len(self._transcript) > 10:
                        self._transcript = self._transcript[-10:]
            elif event == "close":
                self._on_close()

    def _on_close(self):
        global app_running
        self.running  = False
        app_running   = False
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── orb drawing ────────────────────────────────────────────────────────

    def _draw_orb(self):
        W  = self.WIN_W
        H  = self.ORB_AREA_H
        cx = W / 2
        cy = H / 2 - 10
        amp, speed, base_r = self.ORB_SETTINGS.get(self.mode, self.ORB_SETTINGS["standby"])
        color = self.ACCENT.get(self.mode, self.ACCENT["standby"])
        self.phase += speed

        points = []
        for deg in range(0, 360, 5):
            ang  = math.radians(deg)
            wave = amp * (
                0.60 * math.sin(3.0 * ang + self.phase)
                + 0.40 * math.sin(7.0 * ang - 1.3 * self.phase)
            )
            r = base_r + wave
            points.extend([cx + r * math.cos(ang), cy + r * math.sin(ang)])

        c = self.orb_canvas
        c.delete("all")

        # Outer dark ring
        c.create_oval(cx-base_r-18, cy-base_r-18,
                      cx+base_r+18, cy+base_r+18,
                      fill=self.PANEL_BG, outline=self.BORDER, width=1)
        # Waveform
        c.create_polygon(points, outline=color, fill="", width=2, smooth=True)
        # Inner core dot
        c.create_oval(cx-18, cy-18, cx+18, cy+18, fill=color, outline="")

        # Mode text
        c.create_text(cx, cy + base_r + 28,
                      text=f"IRIS  ·  {self.mode.upper()}",
                      fill=color, font=("Courier", 11, "bold"))
        # Caption
        c.create_text(cx, cy + base_r + 50,
                      text=self.caption,
                      fill=self.TEXT_MID, font=("Helvetica", 10))

        # Corner accent lines (Jarvis-style)
        for x0, y0, x1, y1 in [
            (8, 8, 40, 8), (8, 8, 8, 40),
            (W-8, 8, W-40, 8), (W-8, 8, W-8, 40),
            (8, H-8, 40, H-8), (8, H-8, 8, H-40),
            (W-8, H-8, W-40, H-8), (W-8, H-8, W-8, H-40),
        ]:
            c.create_line(x0, y0, x1, y1, fill=self.TEXT_DIM, width=1)

    # ── stats bar ──────────────────────────────────────────────────────────

    def _tick_stats(self):
        if not self.running:
            return
        try:
            cpu  = psutil.cpu_percent(interval=None)
            ram  = psutil.virtual_memory()
            bat  = psutil.sensors_battery()
            now  = datetime.now().strftime("%H:%M:%S")

            bat_str = ""
            if bat:
                plug = "⚡" if bat.power_plugged else "🔋"
                bat_str = f"  {plug}{bat.percent:.0f}%"

            stats = (
                f"  CPU {cpu:4.1f}%  "
                f"RAM {ram.percent:4.1f}%"
                f"{bat_str}  "
                f"🕐 {now}"
            )
            self.stats_label.config(text=stats)
        except Exception:
            pass
        self.root.after(1000, self._tick_stats)

    # ── transcript drawing ─────────────────────────────────────────────────

    def _draw_transcript(self):
        c  = self.trans_canvas
        W  = self.WIN_W
        H  = self.TRANSCRIPT_H
        P  = self.PAD

        c.delete("all")

        # Panel header
        c.create_text(P, 12, text="CONVERSATION", anchor="w",
                      fill=self.TEXT_DIM, font=("Courier", 8, "bold"))
        c.create_line(P, 22, W - P, 22, fill=self.BORDER, width=1)

        with self._transcript_lock:
            lines = list(self._transcript)

        if not lines:
            c.create_text(W // 2, H // 2,
                          text="No conversation yet",
                          fill=self.TEXT_DIM, font=("Helvetica", 10))
            return

        # Draw newest at bottom — walk backwards from bottom of canvas
        y      = H - 10
        max_w  = W - P * 2 - 60

        for role, text in reversed(lines):
            is_iris  = (role == "assistant")
            label    = "IRIS " if is_iris else "YOU  "
            lcolor   = self.ACCENT.get(self.mode, self.ACCENT["standby"]) if is_iris else self.TEXT_MID
            tcolor   = self.TEXT_BRIGHT if is_iris else self.TEXT_MID
            font_lbl = ("Courier",  8, "bold")
            font_txt = ("Helvetica", 9)

            # Word-wrap text
            words = text.split()
            wrapped_lines: list[str] = []
            current = ""
            char_limit = max(20, int(max_w / 5.5))
            for word in words:
                if len(current) + len(word) + 1 <= char_limit:
                    current = (current + " " + word).strip()
                else:
                    if current:
                        wrapped_lines.append(current)
                    current = word
            if current:
                wrapped_lines.append(current)
            wrapped_lines = wrapped_lines[:3]

            line_h = 14
            block_h = len(wrapped_lines) * line_h + 4

            # Draw text lines bottom-up
            for i, wl in enumerate(reversed(wrapped_lines)):
                ty = y - i * line_h
                if ty < 30:
                    break
                c.create_text(P + 52, ty, text=wl, anchor="sw",
                               fill=tcolor, font=font_txt)

            # Draw role label at top of this block
            c.create_text(P, y - (len(wrapped_lines) - 1) * line_h,
                           text=label, anchor="sw",
                           fill=lcolor, font=font_lbl)

            # Separator line between entries
            sep_y = y - block_h + 1
            if sep_y > 28:
                c.create_line(P, sep_y, W - P, sep_y,
                              fill=self.BORDER, width=1, dash=(2, 6))

            y -= block_h + 4
            if y < 30:
                break

    # ── main tick ──────────────────────────────────────────────────────────

    def _tick(self):
        if not self.running:
            return
        self._apply_events()
        self._draw_orb()
        self._draw_transcript()
        self.root.after(30, self._tick)

    def run(self):
        self.root.mainloop()


def ui_mode(mode):
    if gui: gui.post_mode(mode)

def ui_caption(text):
    if gui: gui.post_caption(text)

def ui_transcript(role: str, text: str):
    """Push a new line to the HUD transcript panel."""
    if gui: gui.post_transcript(role, text)


# ──────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────

print("🛠️  Iris starting...")
print(f"Loading Whisper {WHISPER_MODEL_NAME}...")

if not WHISPER_COMPUTE_TYPE:
    WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8"

whisper_model = WhisperModel(
    WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE,
    download_root=str(Path.home() / ".cache/whisper"),
)

try:
    kokoro = Kokoro(str(KOKORO_MODEL_FILE), ensure_kokoro_voice_archive())
    print("✓ Kokoro TTS ready")
except Exception as e:
    logging.warning("Kokoro init failed: %s", e)
    kokoro = None

mem.set_model(MEMORY_MODEL)
memory    = mem.load()
mem.trim_conversation_log()
user_name = memory["user"].get("name") or "sir"

iris_skills.load_skills()

# ── openWakeWord model init ────────────────────────────────────────────────
_oww_model  = None
_oww_lock   = threading.Lock()

def _init_oww():
    global _oww_model
    if not _OWW_AVAILABLE:
        return
    try:
        import openwakeword as _oww_pkg, os as _os
        _models_dir = _os.path.join(_os.path.dirname(_oww_pkg.__file__), "resources", "models")
        _jarvis     = _os.path.join(_models_dir, "hey_jarvis_v0.1.onnx")
        with _oww_lock:
            _oww_model = _OWWModel(wakeword_model_paths=[_jarvis])
        print(f"✓ openWakeWord ready  (model: hey_jarvis proxy, threshold: {OWW_WAKE_THRESHOLD})")
    except Exception as _e:
        logging.warning("openWakeWord init failed: %s — using Whisper fallback.", _e)

threading.Thread(target=_init_oww, daemon=True).start()

print(f"✓ Ready — user: {user_name}, interactions: {memory['interaction_count']}")
print(f"Models → fast: {MODEL_FAST}, smart: {MODEL_SMART}, memory: {MEMORY_MODEL}")
print(f"Whisper → {WHISPER_MODEL_NAME} / {WHISPER_DEVICE} / {WHISPER_COMPUTE_TYPE}")
print(f"Skills  → {len(iris_skills.list_skills())} intents loaded")
print("Say 'Hey Iris' to activate.\n")


def warmup_models():
    for model_name in {MODEL_FAST, MODEL_SMART}:
        try:
            ollama.chat(model=model_name, messages=[{"role": "user", "content": "hi"}])
            logging.info("Warmed up: %s", model_name)
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
                    logging.warning("Kokoro chunk failed, retrying: %s", chunk_error)
                    retry = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", " ", chunk)
                    retry = re.sub(r"\s+", " ", retry).strip()
                    if not retry:
                        continue
                    samples, sample_rate = synthesize_kokoro_audio(retry)
                    sd.play(samples, samplerate=sample_rate)
                    sd.wait()
        else:
            print("[voice]: Kokoro not available")
    except Exception as e:
        print(f"[speak error]: {e}")
    finally:
        time.sleep(0.3)
        is_speaking = False
        if is_processing:
            ui_mode("processing"); ui_caption("Working on it...")
        elif is_active:
            ui_mode("listening"); ui_caption("Listening")
        else:
            ui_mode("standby"); ui_caption("Say 'Hey Iris'")


def quick_transcribe(audio):
    segments, _ = whisper_model.transcribe(
        audio, beam_size=1, language="en", vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
        condition_on_previous_text=False,
    )
    return " ".join(s.text for s in segments).strip().lower().strip(".,!? ")


def is_stop_speech_command(text):
    normalized = re.sub(r"[^a-z\s]", " ", (text or "").lower())
    normalized = " ".join(normalized.split())
    if not normalized or len(normalized.split()) > 3:
        return False
    return any(phrase in normalized for phrase in SPEECH_STOP_PHRASES)


def interrupt_callback(indata, frames, time_info, status):
    if is_speaking:
        interrupt_audio_queue.put(indata.copy())


def interrupt_listener():
    buffer = []
    while app_running:
        if not is_speaking:
            while not interrupt_audio_queue.empty():
                try: interrupt_audio_queue.get_nowait()
                except Exception: pass
            buffer = []
            time.sleep(0.05)
            continue
        try:
            chunk = interrupt_audio_queue.get(timeout=0.5)
            chunk = np.squeeze(chunk)
            buffer.append(chunk)
            if len(buffer) < 6:
                continue
            audio  = np.concatenate(buffer)
            buffer = []
            if np.abs(audio).mean() < SILENCE_THRESHOLD:
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
    last_error = None
    for attempt in range(retries + 1):
        try:
            return ollama.chat(model=model, messages=messages)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(delay * (2 ** attempt))
            else:
                logging.error("Ollama failed after retries for %s: %s", model, e)
    raise last_error


def stream_and_speak(messages, model):
    sentence_buffer = ""
    full_reply      = ""
    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            token = ((chunk or {}).get("message") or {}).get("content", "")
            if not token:
                continue
            sentence_buffer += token
            full_reply      += token
            while True:
                match = re.search(r"(.+?[.!?])(?:\s+|$)", sentence_buffer)
                if not match:
                    break
                sentence       = match.group(1).strip()
                sentence_buffer = sentence_buffer[match.end():].lstrip()
                if len(sentence.split()) > 3:
                    speak(sentence, chunked=False)
        leftover = sentence_buffer.strip()
        if leftover and len(leftover.split()) > 2:
            speak(leftover, chunked=False)
        return full_reply.strip()
    except Exception as e:
        logging.warning("Streaming failed, falling back: %s", e)
        response = chat_with_retry(messages, model=model)
        fallback  = response["message"]["content"].strip()
        if fallback:
            speak(fallback)
        return fallback


def compact_spoken_reply(text, max_sentences=2, max_chars=260):
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(as\s+(an\s+)?ai[^.?!]*[.?!]\s*)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(i\s*(am|'m)\s*phi[^.?!]*[.?!]\s*)", "", cleaned, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    clipped   = [s.strip() for s in sentences if s.strip()][:max_sentences]
    short     = " ".join(clipped) if clipped else cleaned
    if len(short) > max_chars:
        short = short[:max_chars].rsplit(" ", 1)[0].rstrip(".,;: ") + "."
    return short


def split_spoken_chunks(text):
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    chunks = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    return chunks[:3] if chunks else [cleaned]


def synthesize_kokoro_audio(text):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return kokoro.create(text, voice=KOKORO_VOICE, speed=1.05, lang="en-gb", trim=False)


def web_search_answer(query, max_results=3):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None
        return " | ".join((r.get("body") or "")[:200] for r in results[:3] if r.get("body")) or None
    except Exception as e:
        logging.warning("Web search failed: %s", e)
        return None


def needs_web_search(command):
    return bool(set(re.sub(r"[^a-z0-9\s]", " ", command.lower()).split())
                .intersection(REALTIME_KEYWORDS))


def select_chat_model(command):
    normalized = command.lower()
    words      = set(re.sub(r"[^a-z0-9\s]", " ", normalized).split())
    if normalized.startswith(("tell me about", "can you tell me", "what is",
                               "what are", "who is", "explain")):
        return MODEL_SMART
    if words.intersection(SMART_KEYWORDS):
        return MODEL_SMART
    if len(command.split()) > 15:
        return MODEL_SMART
    return MODEL_FAST


def should_skip_history_item(text):
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    normalized = " ".join(normalized.split())
    if not normalized or normalized in FILLER_ONLY_COMMANDS:
        return True
    return any(phrase in normalized for phrase in EXIT_WORDS + SLEEP_WORDS + STOP_PHRASES)


def get_clean_recent_history(memory, n=2):
    history  = mem.get_recent_history(memory, n=10)
    filtered = [h for h in history if not should_skip_history_item(h.get("content", ""))]
    return filtered[-n:]


# ──────────────────────────────────────────────
# WAKE / STANDBY DETECTION
# ──────────────────────────────────────────────

def normalize_for_wake(text):
    return " ".join(re.sub(r"[^a-z\s]", " ", text.lower()).split())

def normalize_command_text(text):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split())

def _contains_phrase(normalized_text, phrase):
    return bool(re.search(rf"\b{re.escape(phrase)}\b", normalized_text))

def command_has_phrase(command_text, phrase):
    return _contains_phrase(normalize_command_text(command_text), phrase)

def is_exit_intent(text):
    norm = normalize_command_text(text)
    exit_phrases = {
        "goodbye", "good bye", "goodbye iris", "good bye iris", "bye", "bye bye",
        "exit", "exit iris", "quit", "shutdown", "shut down", "farewell",
        "terminate yourself", "shut yourself down",
    }
    return any(norm == p or _contains_phrase(norm, p) for p in exit_phrases)

def is_standby_intent(text):
    norm = normalize_command_text(text)
    standby_phrases = {
        "sleep", "sleep iris", "iris sleep", "standby", "stand by",
        "go to standby", "go to sleep", "go to standby mode",
    }
    if any(norm == p or _contains_phrase(norm, p) for p in standby_phrases):
        return True
    words = norm.split()
    for phrase in standby_phrases:
        pw = phrase.split()
        for n in range(max(1, len(pw)-1), min(len(words), len(pw)+1)+1):
            for i in range(len(words)-n+1):
                if difflib.SequenceMatcher(None, " ".join(words[i:i+n]), phrase).ratio() >= 0.78:
                    return True
    return False

def is_wake_word_detected(text):
    normalized = normalize_for_wake(text)
    if not normalized:
        return False
    if WAKE_WORD_RE.search(normalized) or normalized in WAKE_COMMON_MISHEARINGS:
        return True
    words   = normalized.split()
    windows = [" ".join(words[i:i+2]) for i in range(max(0, len(words)-1))]
    windows.append(normalized)
    return any(
        difflib.SequenceMatcher(None, c, t).ratio() >= 0.74
        for c in windows for t in WAKE_TARGETS
    )

def is_daddy_home_detected(text):
    normalized  = normalize_for_wake(text)
    all_targets = {normalize_for_wake(p) for p in DADDY_HOME_PHRASES} | DADDY_HOME_COMMON_MISHEARINGS
    if any(t in normalized for t in all_targets):
        return True
    words   = normalized.split()
    windows = [normalized] + [
        " ".join(words[i:i+n])
        for n in range(4, min(7, len(words)+1))
        for i in range(len(words)-n+1)
    ]
    return any(
        difflib.SequenceMatcher(None, c, t).ratio() >= 0.72
        for c in windows for t in all_targets
    )


# ──────────────────────────────────────────────
# MUSIC (daddy-home scene helper)
# ──────────────────────────────────────────────

def _stop_music_global(silent=False):
    global current_player
    if current_player and current_player.poll() is None:
        try:
            os.killpg(os.getpgid(current_player.pid), signal.SIGTERM)
        except Exception:
            current_player.terminate()
        current_player = None
        if not silent:
            speak("Music stopped.")
    elif not silent:
        speak("Nothing is playing right now.")

def is_music_playing():
    return bool(current_player and current_player.poll() is None)


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
        f"Battery at {battery_str}. CPU at {psutil.cpu_percent()}%."
    )


# ──────────────────────────────────────────────
# DADDY HOME ENTRY
# ──────────────────────────────────────────────

def daddy_home_entry():
    global is_active, current_player
    is_active = True
    ui_mode("listening")
    try:
        if DADDY_HOME_MUSIC.exists():
            _stop_music_global(silent=True)
            current_player = subprocess.Popen(
                ["mpv", "--no-video", "--really-quiet", str(DADDY_HOME_MUSIC)],
                preexec_fn=os.setsid,
            )
    except Exception as e:
        logging.warning("Daddy-home music failed: %s", e)
    time.sleep(1.8)
    name = memory["user"].get("name") or "sir"
    speak(f"Welcome back, {name}. All systems online.")
    time.sleep(0.6)
    _launch_and_snap("code",    side="right")
    time.sleep(1.2)
    _launch_and_snap("firefox", side="left")
    time.sleep(0.8)
    speak("What are we working on today?")


def _launch_and_snap(command, side):
    try:
        if not shutil.which(command):
            return
        subprocess.Popen([command], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        time.sleep(2.5)
        _snap_with_kwin_script(command, side=side)
    except Exception as e:
        logging.warning("launch/snap failed: %s", e)


def _snap_with_kwin_script(command, side):
    qdbus_bin = shutil.which("qdbus6") or shutil.which("qdbus-qt6") or shutil.which("qdbus")
    if not qdbus_bin:
        return False
    target_class = re.sub(r"[^a-z0-9_-]", "", command.lower())
    script_path  = None
    try:
        kwin_script = f"""
var targetClass = "{target_class}";
var side = "{side}";
function snapWindow() {{
    var windows = workspace.windowList();
    var target = null;
    for (var i = windows.length - 1; i >= 0; --i) {{
        var w = windows[i];
        if (!w || !w.normalWindow) continue;
        var rc = (w.resourceClass || "").toLowerCase();
        var rn = (w.resourceName || "").toLowerCase();
        if (rc.indexOf(targetClass) !== -1 || rn.indexOf(targetClass) !== -1) {{ target = w; break; }}
    }}
    if (!target) return;
    var area = workspace.clientArea(KWin.MaximizeArea, target);
    var half = Math.floor(area.width / 2);
    var x = (side === "right") ? area.x + half : area.x;
    target.frameGeometry = {{ x: x, y: area.y, width: half, height: area.height }};
}}
snapWindow();""".strip()
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
            tmp.write(kwin_script)
            script_path = tmp.name
        load = subprocess.run(
            [qdbus_bin, "org.kde.KWin", "/Scripting",
             "org.kde.kwin.Scripting.loadScript", script_path,
             f"iris_snap_{target_class}_{int(time.time()*1000)}"],
            timeout=6, check=True, text=True, capture_output=True,
        )
        sid = re.search(r"\d+", load.stdout or "")
        if not sid:
            return False
        subprocess.run(
            [qdbus_bin, "org.kde.KWin", f"/Scripting/Script{sid.group(0)}",
             "org.kde.kwin.Script.run"],
            timeout=6, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        logging.warning("KWin snap failed for %s: %s", command, e)
        return False
    finally:
        if script_path:
            try: os.unlink(script_path)
            except Exception: pass


# ──────────────────────────────────────────────
# REMINDERS / PROACTIVE
# ──────────────────────────────────────────────

def reminder_checker():
    while app_running:
        try:
            pending = mem.get_pending_reminders(memory)
            if pending:
                mem.mark_reminders_fired(memory, pending)
                for reminder in pending:
                    text = reminder.get("text", "something")
                    waited = 0
                    while (is_speaking or is_processing) and waited < 30:
                        time.sleep(1); waited += 1
                    speak(f"Reminder: {text}.")
                    if not is_active:
                        ui_mode("standby"); ui_caption("Say 'Hey Iris'")
        except Exception as e:
            logging.warning("Reminder checker error: %s", e)
        has_pending = any(not r.get("fired") for r in memory.get("reminders", []))
        time.sleep(30 if has_pending else 120)


def proactive_monitor():
    last_morning  = None
    last_battery_warn = None
    while app_running:
        try:
            if not is_active or is_speaking or is_processing:
                time.sleep(15); continue
            now = datetime.now()
            if now.hour in (7, 8, 9) and last_morning != now.date():
                last_morning = now.date()
                battery = psutil.sensors_battery()
                name    = memory["user"].get("name") or "sir"
                msg     = f"Good morning, {name}."
                if battery:
                    msg += (f" Battery at {battery.percent:.0f} percent"
                            + (" and charging." if battery.power_plugged else ", not plugged in."))
                time.sleep(2); speak(msg)
            battery = psutil.sensors_battery()
            if (battery and not battery.power_plugged and battery.percent < 20
                    and last_battery_warn != now.date()):
                last_battery_warn = now.date()
                time.sleep(1)
                speak(f"Heads up, battery at {battery.percent:.0f} percent. Plug in soon.")
        except Exception as e:
            logging.warning("Proactive monitor error: %s", e)
        time.sleep(60)


# ──────────────────────────────────────────────
# SHUTDOWN
# ──────────────────────────────────────────────

def shutdown():
    global app_running
    app_running = False
    _stop_music_global(silent=True)
    speak("Goodbye sir.")
    if gui: gui.request_close()
    os._exit(0)


# ──────────────────────────────────────────────
# SKILL CONTEXT BUILDER
# ──────────────────────────────────────────────

def _build_skill_context():
    """Build the context dict passed to every skill handler."""
    def _set_active(value: bool):
        global is_active
        is_active = value

    return {
        "memory":    memory,
        "speak":     speak,
        "user_name": user_name,
        "is_active": is_active,
        "set_active": _set_active,
        "shutdown":  shutdown,
    }


# ──────────────────────────────────────────────
# RECORDING / TRANSCRIPTION
# ──────────────────────────────────────────────

def record_until_silence(
    max_duration=10,
    silence_duration=SILENCE_DURATION,
    wait_for_speech_timeout=None,
    hard_max_duration=None,
    max_after_speech_duration=None,
):
    fs              = 16000
    chunk_size      = int(fs * 0.1)
    max_chunks      = int(max_duration / 0.1) if max_duration is not None else None
    wait_chunks     = int(wait_for_speech_timeout / 0.1) if wait_for_speech_timeout is not None else None
    hard_max_chunks = int(hard_max_duration / 0.1) if hard_max_duration is not None else None
    after_speech_chunks = int(max_after_speech_duration / 0.1) if max_after_speech_duration is not None else None
    need_silence    = max(1, int(silence_duration / 0.1))
    audio_chunks    = []
    silence_count   = 0
    started_speaking = False
    chunk_index     = 0
    speech_start_chunk = None
    ambient_samples = []
    dynamic_threshold = SILENCE_THRESHOLD

    with sd.InputStream(samplerate=fs, channels=1, dtype=np.float32) as stream:
        while True:
            if max_chunks is not None and chunk_index >= max_chunks: break
            if hard_max_chunks is not None and chunk_index >= hard_max_chunks: break
            chunk_index += 1
            try:
                chunk, _ = stream.read(chunk_size)
            except Exception as e:
                logging.warning("Audio stream read failed: %s", e); break

            chunk  = np.squeeze(chunk)
            volume = np.abs(chunk).mean()

            if not started_speaking and len(ambient_samples) < 20:
                ambient_samples.append(float(volume))
                ambient_floor     = sum(ambient_samples) / len(ambient_samples)
                dynamic_threshold = max(SILENCE_THRESHOLD, ambient_floor * 2.6)

            if is_speaking: continue

            if volume > dynamic_threshold:
                audio_chunks.append(chunk)
                started_speaking = True
                silence_count    = 0
                if speech_start_chunk is None:
                    speech_start_chunk = chunk_index
            elif started_speaking:
                audio_chunks.append(chunk)
                silence_threshold = max(SILENCE_THRESHOLD * 0.7, dynamic_threshold * 0.82)
                silence_count = silence_count + 1 if volume < silence_threshold else 0
                if silence_count >= need_silence: break

            if not started_speaking and wait_chunks is not None and chunk_index >= wait_chunks:
                break
            if (started_speaking and speech_start_chunk is not None
                    and after_speech_chunks is not None
                    and (chunk_index - speech_start_chunk) >= after_speech_chunks):
                break

    if not audio_chunks:
        return np.zeros(fs, dtype=np.float32), fs
    return np.concatenate(audio_chunks), fs


def transcribe(audio):
    # ── noise reduction ────────────────────────────────────────────────────
    if _NOISEREDUCE_AVAILABLE and audio is not None and len(audio) > 0:
        try:
            audio = _nr.reduce_noise(
                y=audio,
                sr=16000,
                stationary=True,
                prop_decrease=NR_PROP_DECREASE,
            )
        except Exception as _nr_err:
            logging.debug("noisereduce skipped: %s", _nr_err)

    segments, _ = whisper_model.transcribe(
        audio, beam_size=1, language="en", vad_filter=True,
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


# ──────────────────────────────────────────────
# EXECUTE COMMAND
# ──────────────────────────────────────────────

def execute_command(command):
    global memory, user_name, is_active, is_processing

    is_processing = True
    ui_mode("processing")
    ui_caption("Working on it...")

    threading.Thread(target=mem.extract_and_save, args=(memory, command), daemon=True).start()
    mem.add_to_history(memory, "user", command)
    user_name = memory["user"].get("name") or "sir"
    normalized = normalize_command_text(command)

    try:
        # ── filler guard ──────────────────────────────────────────────────
        if normalized in FILLER_ONLY_COMMANDS:
            speak("I heard you. Ask me a full question and I'll answer properly.")
            return

        # ── music bleed guard ─────────────────────────────────────────────
        if is_music_playing():
            safe = (
                is_exit_intent(normalized)
                or is_standby_intent(normalized)
                or any(command_has_phrase(normalized, p) for p in STOP_PHRASES)
            )
            if not safe:
                logging.info("Ignoring likely music bleed: %s", normalized)
                return

        # ── skill dispatch ────────────────────────────────────────────────
        ctx = _build_skill_context()
        handled, reply = iris_skills.dispatch(command, ctx)

        if handled:
            if reply:
                speak(reply)
                ui_transcript("user",      command)
                ui_transcript("assistant", reply)
            mem.extract_extended_memory(memory, command, reply or "")
            mem.log_conversation(command, reply or "")
            if reply:
                mem.add_to_history(memory, "assistant", reply)
            return

        # ── LLM fallback ──────────────────────────────────────────────────
        messages = [{"role": "system", "content": mem.build_system_prompt(memory)}]
        messages.append({
            "role": "system",
            "content": (
                "Stay focused on the user's latest request. "
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
                    "content": f"Real-time web data: {search_data}\nUse this when relevant.",
                })

        messages += get_clean_recent_history(memory, n=6)
        messages.append({"role": "user", "content": command})

        selected_model = select_chat_model(command)
        logging.info("LLM model: %s", selected_model)
        reply = stream_and_speak(messages, selected_model)
        reply = compact_spoken_reply(reply)
        if reply and reply[-1] not in ".!?":
            reply += "."

        ui_transcript("user",      command)
        ui_transcript("assistant", reply)
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
            ui_mode("listening"); ui_caption("Listening")


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# OPENWAKEWORD STANDBY LISTENER
# ──────────────────────────────────────────────

def _oww_standby_listen() -> str:
    """
    Stream mic in 80ms chunks through openWakeWord.
    Returns one of: "wake" | "daddy" | "exit" | "none"
    """
    fs          = OWW_SAMPLE_RATE
    chunk       = OWW_CHUNK_SAMPLES
    pre_buffer  = []
    pre_max     = int(1.5 * fs / chunk)

    CONFIRM_FRAMES = 2
    wake_streak    = 0

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                            blocksize=chunk) as stream:
            while app_running and not is_active:
                if is_speaking:
                    time.sleep(0.05)
                    continue

                raw, _ = stream.read(chunk)
                raw    = np.squeeze(raw)

                pre_buffer.append(raw.copy())
                if len(pre_buffer) > pre_max:
                    pre_buffer.pop(0)

                with _oww_lock:
                    if _oww_model is None:
                        return "none"
                    scores = _oww_model.predict(raw)

                best_score = max(scores.values()) if scores else 0.0

                # ── daddy-home: lower threshold, check via Whisper ────────
                if best_score >= OWW_DADDY_THRESHOLD:
                    post_chunks = int(OWW_POST_WAKE_RECORD_S * fs / chunk)
                    post_buf    = list(pre_buffer)
                    for _ in range(post_chunks):
                        try:
                            extra, _ = stream.read(chunk)
                            post_buf.append(np.squeeze(extra))
                        except Exception:
                            break

                    verify_audio = np.concatenate(post_buf).astype(np.float32)
                    heard        = quick_transcribe(verify_audio)
                    print(f"[OWW trigger {best_score:.2f}] Whisper heard: '{heard}'")

                    if heard and is_exit_intent(heard):
                        return "exit"
                    if heard and is_daddy_home_detected(heard):
                        return "daddy"

                # ── wake: require CONFIRM_FRAMES consecutive hits ─────────
                if best_score >= OWW_WAKE_THRESHOLD:
                    wake_streak += 1
                    if wake_streak >= CONFIRM_FRAMES:
                        print(f"[OWW] Wake confirmed  score={best_score:.3f}")
                        with _oww_lock:
                            if _oww_model:
                                _oww_model.reset()
                        wake_streak = 0
                        return "wake"
                else:
                    wake_streak = 0

    except Exception as exc:
        logging.warning("OWW standby listener error: %s", exc)

    return "none"


def run_assistant_loop():
    global is_active

    try:
        while app_running:
            if is_speaking:
                time.sleep(0.05); continue

            if not is_active:
                print("👂 Standby (openWakeWord)..." if _oww_model else "👂 Standby (Whisper)...")
                ui_mode("standby"); ui_caption("Say 'Hey Iris'")

                # ── OWW streaming standby ─────────────────────────────────
                if _oww_model:
                    _woke = _oww_standby_listen()
                    if _woke == "wake":
                        is_active = True
                        ui_mode("listening"); ui_caption("Listening")
                        name = memory["user"].get("name")
                        if memory["interaction_count"] == 0:
                            greeting = "Iris online. What's your name?"
                        elif name:
                            greeting = f"Online. Good to have you back, {name}."
                        else:
                            greeting = "Online. What do you need?"
                        speak(greeting, chunked=False)
                    elif _woke == "daddy":
                        threading.Thread(target=daddy_home_entry, daemon=True).start()
                    elif _woke == "exit":
                        shutdown()
                    continue

                # ── Whisper fallback standby (OWW not available) ──────────
                audio, _ = record_until_silence(max_duration=None)
                text = transcribe(audio)
                if not text: continue

                if is_exit_intent(text):
                    shutdown()
                elif is_daddy_home_detected(text):
                    threading.Thread(target=daddy_home_entry, daemon=True).start()
                elif is_wake_word_detected(text):
                    is_active = True
                    ui_mode("listening"); ui_caption("Listening")
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
                ui_mode("listening"); ui_caption("Listening")
                audio, _ = record_until_silence(
                    max_duration=None,
                    silence_duration=ACTIVE_SILENCE_DURATION,
                    wait_for_speech_timeout=ACTIVE_WAIT_TIMEOUT,
                    hard_max_duration=ACTIVE_HARD_MAX_DURATION,
                    max_after_speech_duration=ACTIVE_MAX_AFTER_SPEECH_DURATION,
                )
                if audio.size == 0 or float(np.max(np.abs(audio))) < (SILENCE_THRESHOLD * 0.8):
                    continue
                text = transcribe(audio)
                if not text: continue

                command = WAKE_WORD_RE.sub("", text).strip(" .,?!")
                if command:
                    print(f"[Command]: {command}")
                    ui_caption(command)
                    execute_command(command)

    except KeyboardInterrupt:
        shutdown()


def main():
    global gui
    gui_env      = os.getenv("IRIS_GUI", "1").lower()
    gui_enabled  = gui_env not in {"0", "false", "no"}
    has_display  = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    tk_available = tk is not None
    use_gui      = tk_available and gui_enabled and has_display

    if use_gui:
        gui    = IrisVisualizer()
        worker = threading.Thread(target=run_assistant_loop, daemon=True)
        worker.start()
        threading.Thread(target=reminder_checker,  daemon=True).start()
        threading.Thread(target=proactive_monitor, daemon=True).start()
        gui.run()
    else:
        if not gui_enabled:   print("[GUI] Disabled by IRIS_GUI setting.")
        elif not tk_available: print("[GUI] Tkinter not available.")
        elif not has_display:  print("[GUI] No display detected, headless mode.")
        threading.Thread(target=reminder_checker,  daemon=True).start()
        threading.Thread(target=proactive_monitor, daemon=True).start()
        run_assistant_loop()


if __name__ == "__main__":
    main()
