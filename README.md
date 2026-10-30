# 🎙 EchoMate — Your personal offline AI voice companion

Speak → get text → get a smart answer. **Everything runs on your PC.**
Speech-to-text: Faster-Whisper. Answers: Ollama (Llama 3.2). No cloud APIs, no uploads.

## Requirements
- Windows 10/11, a working microphone
- Python 3.10+ (tick **"Add Python to PATH"** while installing)
- ~8 GB RAM recommended

## 1. Install Ollama and the local LLM
1. Download and install Ollama from https://ollama.com/download
2. Open Command Prompt and run:
   ```
   ollama pull llama3.2
   ```
   (Low RAM? Try `ollama pull llama3.2:1b`. EchoMate auto-detects whatever you installed.)
3. Ollama runs in the background (tray icon). If not, run `ollama serve`.

## 2. Set up EchoMate
```
cd EchoMate
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Whisper model
Nothing to install manually: the `base` model (~150 MB) downloads automatically
the **first time you transcribe** (internet needed once). After that it's fully offline.
Want better accuracy? Set `MODEL_SIZE = "small"` in `services/speech_to_text.py`.

## 4. Run
```
python main.py
```
1. Click **Start Listening**, speak, click **Stop Listening**.
2. Your text appears (you can edit it).
3. Click **Suggest Answer** and watch it type out. **Copy Answer** to copy.

## Project structure
```
main.py                     app entry point
ui/main_window.py           GUI + background worker threads
services/audio_recorder.py  microphone capture (sounddevice)
services/speech_to_text.py  Faster-Whisper wrapper
services/llm_service.py     Ollama streaming
```

## Troubleshooting
| Problem | Fix |
|---|---|
| Badge says **Local AI Offline** | Start Ollama (open the app or `ollama serve`) |
| **No model installed** | Run `ollama pull llama3.2` |
| Can't reach microphone | Windows Settings → Privacy → Microphone → allow desktop apps; check default input device |
| "Couldn't load speech model" | Connect to the internet once so the Whisper model can download |
| Transcription empty | Speak closer to the mic / longer than a second |
| Slow answers | Use `llama3.2:1b`, close heavy apps |
| `pip` build errors | Update pip: `python -m pip install --upgrade pip` |

Temporary recordings are saved as WAV files in your temp folder and deleted right after transcription.
