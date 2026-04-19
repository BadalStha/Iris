import os
import subprocess

print("🛠️  Iris JARVIS core online — Gemma 4 + full local mode.")
print("RTX 3050 acceleration active.")
print("Type your command or 'exit' to power down.\n")

while True:
    try:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye", "shutdown", "power down"]:
            print("Iris: All systems shutting down. Good night, sir.")
            break
            
        if not user_input:
            continue

        print("Iris: At your service, sir...")

        # Launch interactive Open Interpreter with correct flags
        cmd = [
            "interpreter",
            "--local",
            "--model", "ollama/gemma4:e4b",
            "--api_base", "http://localhost:11434/v1",
            "--api_key", "ollama",
            "-y"   # Auto-approve safe actions (remove this later if you want manual confirmation every time)
        ]

        subprocess.run(cmd, check=False)

    except KeyboardInterrupt:
        print("\n\nIris: Systems standing by, sir.")
        break
    except Exception as e:
        print(f"Iris: Minor systems glitch: {e}")
