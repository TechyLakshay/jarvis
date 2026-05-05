"""
Jarvis MVP — Voice Assistant
Stack: Whisper (STT) + Gemini 2.0 Flash (brain) + pyttsx3 (TTS)

Install deps:
    pip install openai-whisper pyaudio google-generativeai pyttsx3 pyautogui

On Windows for pyaudio:
    pip install pipwin && pipwin install pyaudio

Set your Gemini API key (free at aistudio.google.com):
    export GEMINI_API_KEY="your_key_here"   (Mac/Linux)
    set GEMINI_API_KEY=your_key_here        (Windows)
"""

import os
import json
import wave
import struct
import tempfile
import webbrowser
import subprocess
import pyaudio
import pyttsx3
import google.generativeai as genai

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "YOUR_KEY_HERE")
WHISPER_MODEL   = "tiny"      # tiny | base | small  (tiny = fastest, ~1s)
SAMPLE_RATE     = 16000
CHUNK           = 1024
SILENCE_LIMIT   = 2.0         # seconds of silence before stopping recording
SILENCE_THRESH  = 300         # amplitude threshold — lower if mic is quiet (run debug_mic.py)
MIC_INDEX       = None        # None = system default. Set to int if wrong mic is used.

# ─── Setup ────────────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)
gemini  = genai.GenerativeModel("gemini-2.0-flash")
tts     = pyttsx3.init()
tts.setProperty("rate", 175)   # speaking speed (words per minute)
tts.setProperty("volume", 1.0)

# Lazy whisper loader — downloads/loads on first transcription call, not at startup
_stt = None

def get_stt():
    global _stt
    if _stt is None:
        print(f"[Jarvis] Loading Whisper '{WHISPER_MODEL}' model (first run only)...")
        try:
            import whisper as _whisper
            _stt = _whisper.load_model(WHISPER_MODEL)
        except Exception as e:
            raise RuntimeError(
                f"Could not load Whisper model: {e}\n"
                "Make sure you ran:  pip install openai-whisper"
            ) from e
    return _stt

# ─── System prompt ────────────────────────────────────────────────────────────
# Jarvis reads this every time. Add new actions here as you build them out.

SYSTEM_PROMPT = """
You are Jarvis, a smart and concise voice assistant.
Given a voice command, respond ONLY with a valid JSON object — no markdown, no explanation.

Available actions:
  open_youtube       params: { query: string }         — search & play on YouTube
  open_website       params: { url: string }           — open any URL in browser
  web_search         params: { query: string }         — google search
  open_app           params: { app_name: string }      — open installed application
  say                params: { text: string }          — just speak a response
  set_volume         params: { level: int (0-100) }    — system volume (Windows only)
  type_text          params: { text: string }          — type text at cursor
  run_command        params: { command: string }       — run a shell command

Response format (strict JSON, nothing else):
{
  "action": "<action_name>",
  "params": { ... },
  "speak": "<what to say before/after doing the action>"
}

Examples:
  "open youtube and play lofi music"
  → {"action":"open_youtube","params":{"query":"lofi music"},"speak":"Opening lofi music on YouTube."}

  "what is the capital of Japan"
  → {"action":"say","params":{"text":"The capital of Japan is Tokyo."},"speak":"The capital of Japan is Tokyo."}

  "open notepad"
  → {"action":"open_app","params":{"app_name":"notepad"},"speak":"Opening Notepad."}
"""

# ─── Recording ────────────────────────────────────────────────────────────────

def record_audio() -> str:
    """Record from mic until silence, save to temp WAV, return path."""
    audio  = pyaudio.PyAudio()
    kwargs = dict(format=pyaudio.paInt16, channels=1,
                  rate=SAMPLE_RATE, input=True,
                  frames_per_buffer=CHUNK)
    if MIC_INDEX is not None:
        kwargs["input_device_index"] = MIC_INDEX

    stream = audio.open(**kwargs)

    print("[Jarvis] Listening... (speak now)")
    frames        = []
    silent_chunks = 0
    max_silent    = int(SILENCE_LIMIT * SAMPLE_RATE / CHUNK)
    max_wait      = int(10 * SAMPLE_RATE / CHUNK)  # 10s hard timeout waiting for speech
    started       = False
    waited        = 0

    while True:
        data      = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        samples   = struct.unpack(f"{CHUNK}h", data)
        amplitude = max(abs(s) for s in samples)  # use abs — handles negative peaks too

        if not started:
            waited += 1
            if amplitude > SILENCE_THRESH:
                started = True
                print("[Jarvis] Got you, recording...")
            elif waited >= max_wait:
                print("[Jarvis] No speech detected. Try speaking louder or run debug_mic.py")
                break
        else:
            if amplitude > SILENCE_THRESH:
                silent_chunks = 0
            else:
                silent_chunks += 1
                if silent_chunks >= max_silent:
                    break  # speech ended

    stream.stop_stream()
    stream.close()
    audio.terminate()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    return tmp.name

# ─── STT ──────────────────────────────────────────────────────────────────────

def transcribe(wav_path: str) -> str:
    """Transcribe WAV file to text using Whisper."""
    print("[Jarvis] Transcribing...")
    result = get_stt().transcribe(wav_path, fp16=False, language="en")
    os.unlink(wav_path)  # clean up temp file
    text = result["text"].strip()
    print(f"[You]    {text}")
    return text

# ─── LLM intent parsing ───────────────────────────────────────────────────────

def parse_intent(command: str) -> dict:
    """Send command to Gemini, get back structured action JSON."""
    print("[Jarvis] Thinking...")
    prompt   = f"{SYSTEM_PROMPT}\n\nUser command: {command}"
    response = gemini.generate_content(prompt)
    raw      = response.text.strip()

    # Strip markdown code fences if Gemini wraps in ```json
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: just say the response
        return {"action": "say", "params": {"text": raw}, "speak": raw}

# ─── TTS ──────────────────────────────────────────────────────────────────────

def speak(text: str):
    """Speak text aloud."""
    print(f"[Jarvis] {text}")
    tts.say(text)
    tts.runAndWait()

# ─── Action executor ──────────────────────────────────────────────────────────

def execute(intent: dict):
    """Run the action Gemini decided on."""
    action  = intent.get("action", "say")
    params  = intent.get("params", {})
    message = intent.get("speak", "")

    if message:
        speak(message)

    if action == "open_youtube":
        query = params.get("query", "")
        url   = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(url)

    elif action == "open_website":
        url = params.get("url", "https://google.com")
        if not url.startswith("http"):
            url = "https://" + url
        webbrowser.open(url)

    elif action == "web_search":
        query = params.get("query", "")
        webbrowser.open(f"https://google.com/search?q={query.replace(' ', '+')}")

    elif action == "open_app":
        app = params.get("app_name", "")
        try:
            if os.name == "nt":                    # Windows
                subprocess.Popen(["start", app], shell=True)
            elif os.name == "posix":               # Mac / Linux
                subprocess.Popen(["open" if "darwin" in os.sys.platform else "xdg-open", app])
        except Exception as e:
            speak(f"Could not open {app}. {e}")

    elif action == "set_volume":
        level = params.get("level", 50)
        try:
            if os.name == "nt":
                # Windows — requires pycaw: pip install pycaw
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(level / 100, None)
            else:
                subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{level}%"])
        except Exception as e:
            speak(f"Volume control failed: {e}")

    elif action == "type_text":
        import pyautogui
        pyautogui.typewrite(params.get("text", ""), interval=0.05)

    elif action == "run_command":
        cmd = params.get("command", "")
        try:
            result = subprocess.check_output(cmd, shell=True, text=True, timeout=10)
            speak(result[:300])  # speak first 300 chars of output
        except Exception as e:
            speak(f"Command failed: {e}")

    elif action == "say":
        # Already spoken above via message; params.text is a fallback
        if not message:
            speak(params.get("text", "I'm not sure how to help with that."))

    else:
        speak(f"I don't know how to do that yet: {action}")

# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    speak("Jarvis online. How can I help?")

    while True:
        try:
            wav_path = record_audio()
            command  = transcribe(wav_path)

            if not command:
                continue

            # Exit commands
            if any(word in command.lower() for word in ["goodbye", "shut down", "exit", "quit"]):
                speak("Goodbye.")
                break

            intent = parse_intent(command)
            execute(intent)

        except KeyboardInterrupt:
            speak("Shutting down.")
            break
        except Exception as e:
            print(f"[Error] {e}")
            speak("Something went wrong. Please try again.")

if __name__ == "__main__":
    main()