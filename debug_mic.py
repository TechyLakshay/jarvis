"""
Mic debugger — run this before jarvis.py to find your mic issues.
    python debug_mic.py
"""

import struct
import pyaudio

CHUNK       = 1024
SAMPLE_RATE = 16000
TEST_SECS   = 5

p = pyaudio.PyAudio()

# ── 1. List all input devices ─────────────────────────────────────────────────
print("\n── Available audio input devices ──────────────────────────────")
input_devices = []
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        input_devices.append(i)
        marker = " ◄ default" if i == p.get_default_input_device_info()["index"] else ""
        print(f"  [{i}] {info['name']}{marker}")

if not input_devices:
    print("  ERROR: No input devices found at all!")
    print("  → Check if your mic is plugged in and not disabled in system settings.")
    p.terminate()
    exit(1)

default_idx = p.get_default_input_device_info()["index"]
print(f"\nDefault device index: {default_idx}")

# ── 2. Try opening the default mic ────────────────────────────────────────────
print("\n── Opening mic stream ─────────────────────────────────────────")
try:
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=default_idx,
        frames_per_buffer=CHUNK
    )
    print("  OK — mic opened successfully")
except Exception as e:
    print(f"  ERROR opening default mic: {e}")
    print("\n  Try a different device index by setting MIC_INDEX below.")
    p.terminate()
    exit(1)

# ── 3. Read audio and show live amplitude ─────────────────────────────────────
print(f"\n── Live amplitude test ({TEST_SECS}s) — speak now! ───────────────")
print("  Quiet   <100  |  Normal speech  500-2000  |  Loud  >5000\n")

frames_read = int(SAMPLE_RATE / CHUNK * TEST_SECS)
max_seen    = 0

for i in range(frames_read):
    try:
        data      = stream.read(CHUNK, exception_on_overflow=False)
        samples   = struct.unpack(f"{CHUNK}h", data)
        amplitude = max(abs(s) for s in samples)
        max_seen  = max(max_seen, amplitude)

        bar   = "█" * min(40, amplitude // 200)
        label = f"{amplitude:5d}  {bar}"
        print(f"\r  {label:<50}", end="", flush=True)
    except Exception as e:
        print(f"\n  ERROR reading audio: {e}")
        break

stream.stop_stream()
stream.close()
p.terminate()

# ── 4. Diagnosis ──────────────────────────────────────────────────────────────
print(f"\n\n── Result ─────────────────────────────────────────────────────")
print(f"  Peak amplitude seen: {max_seen}")

if max_seen == 0:
    print("  PROBLEM: Amplitude is always 0.")
    print("  → Mic is opening but capturing silence.")
    print("  → Try: check mic isn't muted in OS sound settings.")
    print("  → Try: set a different MIC_INDEX from the list above.")
elif max_seen < 100:
    print("  PROBLEM: Amplitude is very low (mic may be muted or gain too low).")
    print("  → Go to sound settings and increase mic input volume.")
elif max_seen < 500:
    print("  WARNING: Low amplitude. Jarvis silence threshold (500) may cut you off.")
    print("  → In jarvis.py, lower SILENCE_THRESH to 100 or 200.")
    print("  → Or increase your mic gain in OS sound settings.")
else:
    print("  OK — mic is working. Peak amplitude is healthy.")
    print("  → If Jarvis still doesn't capture, lower SILENCE_THRESH in jarvis.py.")

print()
print("── Fix guide ──────────────────────────────────────────────────")
print("  If you need a specific mic index, add this to jarvis.py setup:")
print("  MIC_INDEX = 1  # ← change to your device index from the list above")
print("  And in record_audio(), add:  input_device_index=MIC_INDEX")