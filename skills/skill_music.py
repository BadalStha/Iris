"""
skill_music.py — Music / YouTube playback
Handles: play X, stop music, pause.
Priority 20.
"""

import os
import re
import signal
import subprocess

METADATA = {
    "name":        "Music",
    "version":     "1.0",
    "description": "YouTube music playback via mpv",
    "author":      "iris",
}

_current_player = None

PLAY_PHRASES  = ["play", "youtube", "song", "music"]
STOP_PHRASES  = ["stop", "pause", "shut up", "quiet", "mute", "silence",
                 "stop music", "stop the music", "stop playing"]


def _is_playing():
    return bool(_current_player and _current_player.poll() is None)


def _stop(silent=False):
    global _current_player
    if _current_player and _current_player.poll() is None:
        try:
            os.killpg(os.getpgid(_current_player.pid), signal.SIGTERM)
        except Exception:
            _current_player.terminate()
        _current_player = None
        return True
    return False


def _match_play(norm):
    return any(p in norm for p in PLAY_PHRASES)


def _match_stop(norm):
    return any(p in norm for p in STOP_PHRASES)


def _handle_play(command, ctx):
    global _current_player
    query = re.sub(
        r"\b(play|song|on youtube|youtube|music|can you|please)\b",
        " ", command, flags=re.IGNORECASE,
    ).strip(" .,?!")
    if not query:
        ctx["speak"]("What would you like me to play?")
        return None
    _stop(silent=True)
    try:
        _current_player = subprocess.Popen(
            ["mpv", "--no-video", "--really-quiet", f"ytdl://ytsearch1:{query}"],
            preexec_fn=os.setsid,
        )
        return f"Playing {query}."
    except FileNotFoundError:
        return "mpv is not installed. Run: sudo pacman -S mpv"


def _handle_stop(command, ctx):
    stopped = _stop()
    if stopped:
        return "Music stopped."
    return "Nothing is playing right now."


INTENTS = [
    {"name": "play_music", "priority": 20, "match": _match_play, "handle": _handle_play},
    {"name": "stop_music", "priority": 20, "match": _match_stop, "handle": _handle_stop},
]
