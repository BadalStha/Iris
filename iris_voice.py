import ollama
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyttsx3
import threading
import time
import webbrowser
import subprocess
import os

# ============== IRIS CONFIG ==============
MODEL = "gemma4:e2b"          # Change to gemma4:e4b if you pulled the bigger one
WAKE_WORD = "hey iris"
engine = pyttsx3.init()
engine.setProperty('rate', 170)   # Natural speaking speed


# Load faster-whisper (GPU on RTX 3050)
print("Loading Whisper model on RTX 3050...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print(f"🛠️  Iris v1.0 online — Gemma 4 ({MODEL}) + Voice Mode")
print("Listening for 'Hey Iris'... Speak naturally.\n")

def speak(text):
    print(f"Iris: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    print("🎤 Listening...")
    duration = 4  # seconds
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    audio = np.squeeze(recording)
    segments, _ = whisper_model.transcribe(audio, beam_size=5, language="en")
    text = " ".join(seg.text for seg in segments).strip().lower()
    return text

def execute_command(command):
    try:
        if "youtube" in command or "play" in command:
            speak("Playing on YouTube, sir.")
            query = command.replace("play", "").replace("on youtube", "").strip()
            webbrowser.open(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")
        
        elif "open" in command and "browser" in command or "firefox" in command:
            speak("Opening Firefox, sir.")
            webbrowser.open("https://google.com")
        
        elif "weather" in command or "bharatpur" in command:
            speak("Checking weather in Bharatpur, sir.")
            webbrowser.open("https://google.com/search?q=weather+bharatpur+nepal")
        
        else:
            # Let Gemma 4 decide and respond
            response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': command}])
            reply = response['message']['content']
            speak(reply)
            
    except Exception as e:
        speak(f"Minor systems glitch, sir: {str(e)[:80]}")

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
