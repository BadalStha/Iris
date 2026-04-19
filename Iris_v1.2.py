import ollama
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from kokoro import KPipeline
import threading
import time
import webbrowser
import subprocess
import os
import psutil
from pathlib import Path

# ============== IRIS v1.2 - KOKORO NATURAL GIRL VOICE ==============
MODEL = "gemma4:e4b"
WAKE_WORD = "hey iris"

# Load Kokoro TTS pipeline (natural neural voice)
# Voice options: af_heart, af_bella, af_nova, af_sky, af_sarah
# Try each one and pick your favourite!
pipeline = KPipeline(lang_code='a')
VOICE = 'af_heart'

print("Loading Whisper model...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print(f"🛠️  Iris v1.2 online — Gemma 4 ({MODEL}) + Kokoro Natural Voice ({VOICE})")
print("Listening for 'Hey Iris'... Speak naturally.\n")


def speak(text):
    print(f"Iris: {text}")
    generator = pipeline(text, voice=VOICE, speed=1.0)
    for i, (gs, ps, audio) in enumerate(generator):
        sd.play(np.array(audio), samplerate=24000)
        sd.wait()


def listen():
    print("🎤 Listening...")
    duration = 5
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    audio = np.squeeze(recording)
    segments, _ = whisper_model.transcribe(audio, beam_size=5, language="en")
    text = " ".join(seg.text for seg in segments).strip().lower()
    return text


def play_youtube(query):
    speak(f"Opening {query} on YouTube, sir.")
    search_query = query.replace("play", "").replace("song", "").strip()
    url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
    webbrowser.open(url)


def search_files(query):
    home = Path.home()
    speak("Searching your files, sir.")
    matches = list(home.rglob(f"*{query}*"))[:10]
    if matches:
        speak(f"Found {len(matches)} matches. Opening the first one.")
        subprocess.run(["xdg-open", str(matches[0])])
    else:
        speak("No matching files found, sir.")


def get_system_info():
    battery = psutil.sensors_battery()
    battery_str = (
        f"{battery.percent}% {'charging' if battery.power_plugged else 'on battery'}"
        if battery else "unknown"
    )
    speak(
        f"Current time is {time.strftime('%I:%M %p')}. "
        f"Battery is at {battery_str}. "
        f"CPU usage is {psutil.cpu_percent()}%."
    )


def execute_command(command):
    try:
        if "play" in command or "youtube" in command or "song" in command:
            play_youtube(command)
        elif "find" in command or "search file" in command or "where is" in command:
            query = command.replace("find", "").replace("search file", "").replace("where is", "").strip()
            search_files(query)
        elif "battery" in command or "time" in command or "cpu" in command or "status" in command:
            get_system_info()
        elif "open browser" in command or "google" in command:
            speak("Opening browser, sir.")
            webbrowser.open("https://google.com")
        else:
            response = ollama.chat(
                model=MODEL,
                messages=[{'role': 'user', 'content': f"You are Iris, a helpful AI assistant. {command}"}]
            )
            reply = response['message']['content']
            speak(reply)
    except Exception as e:
        speak(f"Minor glitch, sir: {str(e)[:60]}")


# Main loop
try:
    while True:
        text = listen()
        if WAKE_WORD in text:
            speak("At your service, sir.")
            full_command = text.replace(WAKE_WORD, "").strip()
            if full_command:
                print(f"You said: {full_command}")
                threading.Thread(target=execute_command, args=(full_command,)).start()
            else:
                speak("Yes, sir? What can I do for you?")
except KeyboardInterrupt:
    speak("Shutting down all systems. Good night, sir.")