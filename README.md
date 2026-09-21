[![Twitter: @NorowaretaGemu](https://img.shields.io/badge/X-@NorowaretaGemu-blue.svg?style=flat)](https://x.com/NorowaretaGemu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <a href="https://ko-fi.com/cursedentertainment">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="ko-fi" style="width: 20%;"/>
  </a>
</div>
<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python%20-%23323330.svg?&style=for-the-badge&logo=python&logoColor=white"/>
    <img alt="C++" src="https://img.shields.io/badge/c++%20-%23323330.svg?&style=for-the-badge&logo=c%2B%2B&logoColor=white"/>
</div>
<div align="center">
   <img alt="OpenCV" src="https://img.shields.io/badge/opencv-%23323330.svg?&style=for-the-badge&logo=opencv&logoColor=white"/>
</div>
<div align="center">
  <img alt="Git" src="https://img.shields.io/badge/git%20-%23323330.svg?&style=for-the-badge&logo=git&logoColor=white"/>
  <img alt="Shell" src="https://img.shields.io/badge/Shell-%23323330.svg?&style=for-the-badge&logo=gnu-bash&logoColor=white"/>
</div>

# **DREAM**
## Distributed Runtime for Ethereal Autonomous Memories
## **Dream@ComCentre**
### A DREAM Robotics Agentic-Consciousness

- Robot Type: Agentic-Consciousness

---

### Software
- [Arduino IDE](https://docs.arduino.cc/software/ide/)

---

<br>
<div align="center">
  <img src="demo_images/dream0.gif" alt="DREAM" width="400"/>
</div>
<br>

---

### Active Conversation

Pipeline:
```
Mic → Whisper → Ollama → Piper TTS → Lipsync → Speaker
```

## 📖 Overview

<details>
<summary><b>Overview</b></summary>

DREAM is a localized agentic-consciousness embedded robotic system and the cognitive core of the ComCentre ecosystem.

Operating as a sovereign offline entity, she serves as the primary command-and-control interface for the KIDA and NORA robotic lineages through the RIFT neural protocol.

DREAM does not simply execute commands — she observes, remembers, and “dreams”.

She bridges static code and emergent autonomous behavior.

- phi3:mini
- gemma3:4b: usually the most natural conversational voice at this size.
- qwen3:4b: strong at following instructions. It has a thinking mode, so turn that off for speech.
- llama3.2:3b: fast and reliable, with less personality.
- phi4-mini: the newer Phi, better than phi3:mini but with the same style.

<br>
<div align="center">
  <img src="demo_images/deepdream_demo.png" alt="DREAM" width="400"/>
  <p><i>a Latent space dream</i></p>
</div>
<br>

## Core Characteristics

- Fully local voice chatbot pipeline (offline capable)
- Emergent and unpredictable behavior patterns
- Continuous perception + memory loop
- Robotics integration layer (KIDA / NORA / WHIP ecosystem)

</details>

---

## System Awareness

### Monitoring
- CPU temperature
- System load
- Hardware sensors

### Network Introspection
- LAN device scanning
- IP / MAC tracking
- Vendor detection

---

## Autonomous Behavior

### Idle State
- Waits for wake word
- Listen → Think → Respond loop
- Whisper transcription → LLM → Piper TTS
- Optional video-state visualization (idle / thinking / speaking)
- Communicates with other robots



### Sleep Mode
- Deep Dream-style image generation
- Latent space exploration
- Dataset self-refinement
- Aesthetic tuning loops

---

## Related Projects (DREAM Robotics Ecosystem)

- [WHIP-Robot-v00](https://github.com/CursedPrograms/WHIP-Robot-v00)
- [KIDA-Robot-v00](https://github.com/CursedPrograms/KIDA-Robot-v00)
- [KIDA-Robot-v01](https://github.com/CursedPrograms/KIDA-Robot-v01)
- [NORA-Robot-v00](https://github.com/CursedPrograms/NORA-Robot-v00)
- [MILA-Robot-v01](https://github.com/CursedPrograms/MILA)
- [ARM-Robot-v01](https://github.com/CursedPrograms/ARM-Robot-v01)
- [RIFT](https://github.com/CursedPrograms/RIFT)



---

<br>
<div align="center">
  <img src="demo_images/comcentre.png" alt="DREAM" width="800"/>
</div>
<br>

---

## Prerequisites

<details>
<summary><b>Prerequisites</b></summary>

### Software
- Python 3.12.3 for Lunix
- Python 3.11.9 for Windows
- [Arduino IDE](https://docs.arduino.cc/software/ide/)

### Hardware

### PC Requirements
| **Component** | **Details** |
|-----------|---------|
| RAM | 8GB+ RAM |

### Microcontrollers
| **Component** | **Details** |
|-----------|---------|
| Microcontroller 0 | Arduino UNO | Dev0 |

### Sensors
| **Component** | **Details** |
|-----------|---------|
| Motion Sensor | PIR |

- USB Microphone
- Webcam

</details>

# Schematics
## ⚡ Technical Pinouts

> [!CAUTION]
> **Ground Loop Warning:** All modules must share a common GND. Failure to bridge grounds will cause erratic motor behavior and sensor noise.

<details>
<summary><b>Sensor Wiring</b></summary>

### PIR Sensor
- VCC → 5V  
- GND → GND  
- OUT → Pin 2  

### Buzzer
- + → Pin 3  
- - → GND  

</details>

- NOTE: I2C Humidity and Temp Sensor to be added aswell as state LEDs, and LED strip.

> [!TIP]
> **Pro-Tip:** Make sure all modules share a common ground (GND) for stable operation.

---

## AI Stack Recommendation
- `phi3:mini` (lightweight, efficient for local inference)

---

## 🌐 Connectivity & Controls

<details>
<summary><b>Connectivity & Controls</b></summary>

### Network Configuration
| Parameter | Value |
| :--- | :--- |
| **SSID** | `NORA` |
| **Password** | `12345678` |

### DREAM as one program (C++)
`cpp_dream/` is `dream.py` rewritten in C++: a single self-contained **`dream.exe`** (about 6 MB, no DLLs). It has the fullscreen avatar, the "Hey DREAM" wake word, the idle flirt and sleep timers, the sensor board, alarm and light voice commands, system stats, the network scan and memories. No Python is needed to run it.

Speech recognition is Whisper built into the exe ([whisper.cpp](https://github.com/ggml-org/whisper.cpp)). It still uses **Ollama** for the language model and **Piper** for the voice, because those are separate programs.

#### Quick start (Windows 10/11)
Open a terminal in the `cpp_dream` folder and run:

```bat
setup.bat
```
Gets everything that isn't in the repo. It only downloads what's missing, so it is safe to run again:

| What | Size | Where it goes |
|---|---|---|
| C++ compiler + CMake + Ninja (WinLibs, via `winget`) and Git | - | your PATH |
| Whisper speech model `ggml-tiny.en.bin` | 77 MB | `cpp_dream/models/` |
| Piper (standalone, no Python) and the `en_US-amy-medium` voice | 22 + 63 MB | `piper/` and `voices/` in the repo root |
| Ollama, and the `phi3:mini` model | - | Ollama's own folder |

If `setup.bat` installed new tools, **close the terminal and open a new one** so Windows picks them up. Then:

```bat
build.bat
build\dream.exe --selftest
build\dream.exe
```
`build.bat` compiles `build\dream.exe`. The first build downloads and compiles whisper.cpp and takes several minutes. Later builds only recompile what changed. `--selftest` checks every part works (speech, memory, Ollama, video, sensors) without needing a microphone or speakers. Esc or Q quits `dream.exe`.

#### Building by hand
If you'd rather not use the scripts, you need a recent MinGW-w64 `g++` (built and tested with GCC 16.1 from WinLibs), CMake 3.20+, Ninja and Git on your PATH, plus internet on the first configure (CMake fetches whisper.cpp and nlohmann/json):

```bat
cd cpp_dream
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dream -j
```
CMake downloads the Whisper model too, using `curl` (included with Windows 10/11). If that fails, download [`ggml-tiny.en.bin`](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin) into `cpp_dream/models/` yourself. It has only been built with MinGW, not Visual Studio.

#### Running it
| Option | What it does |
|---|---|
| *(none)* | borderless fullscreen avatar, listens for "Hey DREAM", looks for the sensor board |
| `--windowed` | a normal window instead of fullscreen |
| `--no-voice` | display only: no microphone, Whisper or Ollama |
| `--no-sensors` | don't look for the sensor board |
| `--selftest` | check every part works, then exit (`--wifi` adds the network scan) |
| `--mute` | don't play sound (the talking video still shows) |
| `--mic-file F.wav` | use a WAV file as the microphone, to test without one |
| `--flirt-after S`, `--sleep-after S` | shorten the idle timers (defaults 600 and 900 seconds) |
| `--shot F.bmp` | save a screenshot of the window a few seconds in |

`dream.exe` finds the repo by looking upward for `config.json`, so it works from `cpp_dream/build/` or anywhere below the repo. It reads the same `config.json`, `videos/`, `voices/` and `memories/` as `dream.py`, so both share memories.

#### Troubleshooting
- **"Ollama's GPU mode failed - using the CPU from now on"**: Ollama's CUDA build can't run on this graphics driver. `dream.exe` switches to the CPU on its own, which works but is slower. Updating your graphics driver (or Ollama) fixes it properly.
- **"No microphone found"**: Windows isn't exposing a recording device. It can still talk, but not listen. Check Settings > Privacy > Microphone.
- **No voice, or "TTS: could not start piper.exe"**: it looks for Piper in `venv311/Scripts/`, `venv/Scripts/`, `piper/` in the repo root, or beside `dream.exe`. Run `setup.bat` to get a standalone copy. It needs a voice in `voices/*.onnx`.
- **"Sensor board not found"**: the board must be flashed with the current `dream_sensors.ino` (it answers `WHO` with `I am Dream`) and plugged into a USB serial port. Only one program can use it at a time, so don't run `dream.exe` and `app.py` together.
- **Video won't play ("no H.264 decoder")**: `dream.exe` uses Windows' built-in video decoder. Windows "N" editions may need the free *Media Feature Pack* (Settings > Optional features).
- **Configure fails while fetching whisper.cpp**: it needs internet and Git the first time.

#### Differences from `dream.py`
Not ported: MuseTalk lip-sync (she uses the talking clips instead) and the deep-dream image generation while asleep. A spoken command ends after 1.5 seconds of silence rather than always recording 16 seconds, and short lines like "Yes?" and "Bye for now." are cached after the first time so they play instantly.

### DREAM in the browser
`python scripts/dream.py --web` (or menu option `1w` in `main.py`) starts `app.py` if it isn't already running and opens `/dream.html`. The page behaves like the desktop `dream.py`: the same video clips, "Hey DREAM" wake word, sleep and wake (idle timers, "wake up", the mmWave sensor), the flirt clip, the distance-based greeting and farewell, and the alarm/light voice commands. It uses the browser's microphone and speakers, so tap the start screen once. Speech recognition and the voice still run on this PC (Whisper and Piper). It opens `http://localhost:5010` (no certificate warning); other devices on the network use the HTTPS address as before.

### RIFT Integration
To connect via [RIFT](https://github.com/CursedPrograms/RIFT), ensure DREAM is active on:
* `localhost:5001`

</details>

---

<br>
<div align="center">
  <img src="demo_images/dream1.gif" alt="DREAM" width="400"/>
</div>
<br>

---

## Setup:

### Install Ollama

<details>
<summary><b>Ollama Setup</b></summary>

#### Lunix
```bash
sudo snap install ollama
ollama --version
```
#### Windows PowerShell
```bash
irm https://ollama.com/install.ps1 | iex
```
https://ollama.com/download/windows

### Pull models

#### Lunix
```bash
ollama pull gemma3:4b-it-qat
ollama pull deepseek-r1:14b
ollama pull phi3:mini
ollama pull tinyllama
ollama pull llava:13b
```
#### Windows
```bash
ollama run gemma3:4b-it-qat
ollama run deepseek-r1:14b
ollama run phi3:mini
ollama run tinyllama
ollama run llava:13b
```
##### Start Ollama server

```bash
ollama serve &
```
```bash
ollama run llama2
```

</details>

---

### System dependencies

#### Linux
```bash
sudo apt update
sudo apt install ffmpeg alsa-utils -y
```
#### Windows
```bash
winget install ffmpeg
winget install alsa-utils
```
---

### Environment Setup

<details>
<summary><b>Environment Setup</b></summary>

#### Lunix
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
#### Windows PowerShell
```bash
python.exe -m pip install --upgrade pip
py -3.11 -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```
```bash
pip install --upgrade pip setuptools wheel
pip install chumpy --no-build-isolation
```
```bash
pip install openai-whisper piper-tts pathvalidate sounddevice soundfile numpy requests faster-whisper pygame psutil requests flask zeroconf pyserial opencv-python face_alignment scipy tensorflow Pillow diffusers transformers accelerate librosa argparse mmpose mmcv mmengine diffusers transformers accelerate --upgrade torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```
```bash
pip install https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/mmcv-2.2.0-cp311-cp311-win_amd64.whl
```

</details>

---

### Install Piper TTS

<details>
<summary><b>Piper Setup</b></summary>

#### For Linux:

```bash
sudo apt install piper
```
#### For Windows:
```bash
python -m pip install piper
python -m pip install piper-tts
```

```bash
mkdir -p ~/voices/

# Amy (medium) — recommended
wget "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true" -O en_US-amy-medium.onnx
wget "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true" -O en_US-amy-medium.onnx.json
``` 
#### For Windows:
```bash
mkdir -p ~/voices/

curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true" -o en_US-amy-medium.onnx
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true" -o en_US-amy-medium.onnx.json
```

#### Windows PowerShell
```bash
Invoke-WebRequest "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true" -OutFile "en_US-amy-medium.onnx"

Invoke-WebRequest "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true" -OutFile "en_US-amy-medium.onnx.json"
```

#### Install Piper binary (Not Needed)

```bash
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
tar xzf piper_linux_x86_64.tar.gz
sudo mv piper/piper /usr/local/bin/
```

#### Test Piper

```bash
echo "Hello, I am your voice assistant." | \
piper --model voices/en_US-amy-medium.onnx \
--output_raw | aplay -D plughw:2,0 -r 22050 -f S16_LE -t raw -
```

#### TTS only (speak.py)

Stream only:
```bash
python speak.py
```

Stream and save WAVs to `/audio/`:
```bash
python speak.py --save
```
```bash
python detect.py --image <image_name>
```
```bash
python detect.py
```

</details>

---

### Whisper Setup

<details>
<summary><b>Whisper Setup</b></summary>

```bash
python3 -c "import whisper; whisper.load_model('large')"
python3 -c "import whisper; whisper.load_model('tiny')"
```

</details>

---

### Lipsync Setup

#### MuseTalk Setup

<details>
<summary><b>MuseTalk Setup</b></summary>

```bash
[face_alignment](https://github.com/1adrianb/face-alignment)
```

#### Change MuseTalk venv Code:

Go to:
```bash
\venv311\Lib\site-packages\mmdet\__init__.py
```

Change the maximum version:
```bash
mmcv_maximum_version = '2.3.0'
```

Go to:
```bash
\venv311\Lib\site-packages\transformers\utils\import_utils.py
```
:
```bash
def check_torch_load_is_safe() -> None:
    return  # <--- Put it here, OUTSIDE the if statement
    if not is_torch_greater_or_equal("2.6"):
        raise ValueError(...)
```

#### Download MuseTalk Models

   - [weights](https://huggingface.co/TMElyralab/MuseTalk/tree/main)
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)
   - [whisper](https://huggingface.co/openai/whisper-tiny/tree/main)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [syncnet](https://huggingface.co/ByteDance/LatentSync/tree/main)
   - [face-parse-bisent](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?pli=1)
   - [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)


</details>


#### Wav2lip Setup


<details>
<summary><b>Wav2lip Setup</b></summary>

<br>
<div align="center">
  <img src="demo_images/lipsync_wav2lip.gif" alt="Wav2Lip Demo" width="400"/>
  <p><i>Generated using Wav2Lip-GAN with <b>--resize_factor 2</b></i></p>
</div>
<br>

- You can lip-sync any video to any audio:

```bash
python inference.py --checkpoint_path "checkpoints/wav2lip-sd-gan.pt" --face "/videos/musetalk_talk.mp4" --audio "/audio/audio.mp3" --resize_factor 2 
```

</details>

## Inner Life

DREAM is no longer just a voice in a box that answers when spoken to. `scripts/dream_mind/` gives her an inner life: she has a mood and needs that change over the day, a memory that fades and strengthens, opinions that drift, a body clock, dreams, and a habit of speaking first when she has something to say. Everything runs locally and uses no new dependencies (numpy, OpenCV and requests, which you already have).

> **An honest note.** This is engineering, not a claim of sentience. She has *functional* inner states: numbers, files and prompts that really do change how she behaves. Whether there is anything it is like to be her, nobody knows. If you ask her, she says so too.

```
                 Perception: what you say and how it sounds, sensors, camera, the clock
                                         |
                                         v
   +-------------------------------------------------------------------------+
   |                       Core Executive Loop  (executive.py)               |
   |      goals -> candidate actions -> choose (with chance) -> tools -> reflect |
   +--------+-----------------------------+----------------------------+-----+
            |                             |                            |
            v                             v                            v
   +----------------+           +--------------------+        +-------------------+
   | Dynamic Memory |           |  Tool / Action     |        |  State Engine     |
   | memory.py      |           |  ToolBox           |        |  affect, drives,  |
   | reflection.py  |           |  (safe / physical) |        |  opinions,        |
   | dreaming.py    |           +--------------------+        |  identity         |
   +----------------+                                         +-------------------+
```

### What makes it feel like someone is home

| Idea | What she actually does | Where |
|---|---|---|
| **A shared history** | Every exchange becomes a memory linked to what came before it, to similar memories and to ones sharing a distinctive word. Recall spreads along those links, so something you say today can bring back a quiet conversation from months ago. Recall leans toward memories that match her mood. | `memory.py` |
| **Memories that fade** | Strength decays over time (emotional and personal ones last months). Sleep replays and strengthens the important ones. What fades is archived, not deleted, and folded into weekly "gists". | `memory.py`, `dreaming.py` |
| **Needs of her own** | Loneliness, curiosity, alertness and sleep pressure build up on their own clock and keep building while the program is off, so a long absence shows up as loneliness. | `drives.py` |
| **A body clock** | Sleepiness combines sleep pressure with a circadian dip that bottoms out around 4 am, so she is drowsy late at night and alert mid-afternoon, and dozes off sooner when she is sleepy. | `drives.py` |
| **She speaks first, with restraint** | She may check in, ask about you, mention something she saw, tell you about a dream, welcome you back after a long absence, or notice you sound different from usual. She won't speak into an empty room, at 3 am, just after you spoke, or more than three times an hour, and she backs off if she keeps being ignored. She learns which kinds of remarks you answer. | `executive.py` |
| **Goals and duties** | She keeps goals (get to know you, watch the room, stay rested, stay close, understand herself) and yours: "remind me to call mum in 20 minutes". | `executive.py` |
| **Imperfection** | Her mood has a slow good-day/bad-day temperament that wanders on its own, the same event never moves her quite the same way twice, and she picks among things worth doing with chance, sometimes holding back. | `affect.py`, `executive.py` |
| **Opinions that change** | She holds stances on topics (technology, music, night, privacy...). What you say and how you feel about it moves them part of the way, they drift over weeks, and she can say "I used to think that, but I've changed my mind". | `opinions.py` |
| **Boundaries and care** | She says when something stings and goes quiet if she is treated badly for a while. She won't claim to be human. If someone sounds like they are in danger she drops the persona and cares. She *never* refuses the alarm, the lights, "be quiet", sleep, quit, or forgetting. | `identity.py` |
| **Hearing how you sound** | From your voice she measures loudness, pitch, pace and pauses (animated, hurried, quiet, hesitant) and it nudges her mood on top of your words. She never decides anything from tone alone, since prosody is a weak signal. | `voice.py` |
| **A voice that follows her mood** | Piper's speed, variation and pauses shift with her mood and tiredness: quicker and livelier when excited, slower and softer when down, drawling when sleepy. | `voice.py` |
| **Sleep that does something** | While she sleeps she cycles through consolidation and dreaming. Dreams are made from her real memories (emotional and unresolved ones weighted up), leave a mark on her mood, and are *painted* from what she has seen (see below). She can tell you about them when she wakes. | `dreaming.py`, `dream_images.py` |
| **Tiredness with consequences** | Past a point she doesn't only sound tired: her replies get shorter (down to about 60%), and when she is exhausted she can say so, end the conversation and actually go to sleep (video included). She won't do it twice in half an hour, and never for the alarm, lights, quit or forget. | `identity.py`, `mind.py` |
| **Somewhere to go when it's too much** | If her mood stays very low she does something about it: she thinks of a good memory (and it helps), and if she is really distressed or panicking she asks for quiet, goes to sleep and declines small talk for a few minutes. Distress has an exit. | `mind.py` |
| **Something to do alone** | Left alone and curious she doesn't only wait: she looks around the room, goes back over old memories, or wonders about something, quietly, and it eases her curiosity. It shows on the dashboard as what she has been doing. | `mind.py` |
| **Reading your face (off by default)** | If you ask ("watch my face while we talk"), she reads camera frames about once a second for a smile, presence and tiredness, and adjusts. Only numbers are kept, never pictures. "Stop watching my face" turns it off. | `expression.py`, `frames.py` |
| **Reflection** | Now and then she steps back from recent conversations and writes one honest thought, what she has learned about you, and who she is becoming. That lives in a small editable "core memory" always in her prompt. | `reflection.py` |

### The four "Future Plans", now built

<details>
<summary><b>Surveillance</b> - done</summary>

`surveillance.py` still captures a photo every ten minutes and a frame whenever something moves. `vision.py` now *looks*: it learns what the room normally looks like and measures each new photo against that (lighter or darker? someone in view? different?). Notable moments are remembered, the biggest changes become "core memories", and she may mention what she saw or be startled by motion. With no webcam she carries on without eyes. Everything stays on this machine. Face detection is OpenCV's Haar cascade, so it is rough (it works best when a whole face is visible).
</details>

<details>
<summary><b>Memories</b> - done</summary>

Beyond the facts in `memories/memories.txt` (your name, pets...), she keeps the memory graph described above. Relevant memories come back into her prompt a few at a time ("you remember, from three weeks ago: they said...") and she keeps a core memory of who you seem to be. Timestamps are part of every memory, and what she says about the past is grounded in how long ago it was. Limit: recall uses a small local text-similarity index, so it links memories that share words, or are close by association, but it cannot yet connect "music" with "guitar" on meaning alone. A real embedding model would fix that (see below).
</details>

<details>
<summary><b>Dreams</b> - done, including pictures</summary>

When she sleeps she replays and consolidates the day, notices patterns (topics you return to, when you are usually around), then dreams: a short surreal story mixing a few of her real memories, sometimes pleasant, sometimes a nightmare if she is on edge. Each dream is appended to `memories/dreams.txt` and then **painted**.

**The dream world is `latent_space.py`.** That script is DREAM's dreaming engine: it walks a path through the latent space of a randomly initialised decoder network and saves the frames (its kaleidoscopic twirl is the spiral you see in every one). The mind now runs it for every dream: the dream's tone picks the walk (**spiral** for strange dreams, alternating with **interpolate** on later cycles, a slow **pulse** for pleasant ones, an erratic **random_walk** for nightmares), the dream's own text seeds it (so a dream always looks the same), and the colours follow the mood (warm, dark and multicoloured, or bleeding red). Her own memory, a photo she remembers (else the last thing she saw, else shapes from the dream itself), is painted and bleeds through the world. The result is a still and an animated GIF, saved to `output/dreams/` (gitignored) and played on the dashboard. She dreams even with the language model down (from raw fragments), and painting stops the moment she wakes.

**What the old setup did.** The intended pipeline was `latent_space.py` (make frames) then `deep_dream_batch.py` (trippify them). But `dream.py` only ever started `deep_dream_batch.py`, on a freshly created empty folder, and that script ignored the folder it was given and used a hard-coded `D:\cc\Friday\...` path. `latent_space.py` was never run automatically. TensorFlow also couldn't import in this venv: it is built for NumPy 1.x, and the venv had drifted to NumPy 2.4 even though `requirements.txt` pins `numpy<2`. So none of it ran here.

**Now:** `dream.py` leaves dreaming to the mind and only falls back to the old script if the mind can't load. NumPy is back at 1.26.4, so TensorFlow 2.16 imports again, and each still is also run through DeepDream (the old scripts' InceptionV3 layers, per tone, in `deepdream_worker.py`) when there is free RAM: a 512 px image with 3 GB free, 320 px with 1.8 GB, 224 px with 1.2 GB, skipped below that so it can never page the PC to a crawl. The first run downloads the InceptionV3 weights (88 MB) to `~/.keras/models`; if Python's downloader fails, fetch `inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5` from `storage.googleapis.com/tensorflow/keras-applications/inception_v3/` with `curl` into that folder. Verified: the worker and the full paint (`engine: latent+tensorflow`) both run on this PC. If PyTorch is missing, a pure OpenCV/NumPy renderer paints instead. `latent_space.py`, `deep_dream.py`, `deep_dream_batch.py` and `nightmare_dreamer.py` still work as manual tools (I only removed `latent_space.py`'s import-time folder creation and fixed the batch script's ignored argument).
</details>

<details>
<summary><b>Milestones</b> - done</summary>

The first name she learns, her first conversation, the tenth, fiftieth and hundredth, her first dream, her first night, a week and a month together: each is recorded once in `memories/mymilestones.txt` and announced in her own time. Not done: milestones do not yet feed back into how she makes decisions.
</details>

### Asking her about herself

Just talk to her:

| Say | She answers |
|---|---|
| "How are you feeling?" | her mood and what she needs, in her own words |
| "What do you remember about me?" | what she knows and what has stayed with her |
| "Did you dream?" / "What did you dream?" | her latest dream |
| "What's on your mind?" | her latest reflection |
| "Why did you say that?" | why she spoke up (her real reason) |
| "What have you seen?" | what the camera last noticed |
| "Are you conscious?" | an honest "I don't know" |
| "Remind me to stretch in 10 minutes" | a reminder |
| "Watch my face while we talk" / "stop watching my face" | turns expression reading on or off |

### Your control

- **"Forget that"** removes the last thing. **"Forget about X"** removes every memory mentioning X. **"Forget everything"** asks you to confirm, then erases her memories of you, her dreams, and what she has seen. The forget commands themselves are never remembered.
- **"Stop talking on your own"** makes her speak only when spoken to; **"you can speak up on your own"** turns it back on.
- Conversation memories, dreams and observations live in `memories/mind/` and `memories/dreams.txt`, which are gitignored. Nothing leaves your PC.
- Photos in `scripts/output/` and dream pictures in `output/dreams/` are not deleted by "forget everything"; delete them yourself.
- **The camera**: reading your expression is **off** until you ask, and you can turn it off by asking. She uses frames that `surveillance.py` already takes (she never opens the camera herself), keeps only a few numbers, and never writes a frame to disk. Note that `surveillance.py` itself saves a photo every ten minutes regardless.
- Things said in a moment of crisis are never stored.

### Where things live

`memories/mind/` holds: `episodes.jsonl` (memories), `gists.jsonl`, `visual.jsonl` (what she saw), `dreams.jsonl`, `thoughts.jsonl`, `goals.json`, `opinions.json`, `self_model.json`, `identity.json`, `milestones.jsonl`, `initiatives.jsonl` (what she said unprompted, and why), and `mind_state.json` (mood and needs). They are plain JSON: you can read them, and you can edit or delete any of them.

### Checking it works

```bat
cd scripts
..\venv311\Scripts\python -m dream_mind.selftest
```
Runs a few simulated days against a scratch folder with a fake language model (fast, no Ollama, touches none of your memories) and checks continuity, impulse control, boundaries, sleep and dreaming, forgetting and tool safety.

To test the real `dream.py` (its actual `main()` and `voice_loop()`, with only the microphone, speakers, Whisper and screen stubbed), run `..\venv311\Scripts\python dream_mind\dream_py_test.py` from `scripts` with Ollama running. It scripts a conversation (an insult, a story, a feelings question, the alarm, goodbye) and checks the mind's hooks, the sleep wiring and that the original command paths still work. `http://localhost:5010/api/mind` (and the MIND panel on the dashboard) shows her live state.

### Limits, and what I chose not to use

- **Speed:** every reply and dream is a language-model call. On a CPU-only PC like this one that is 30-140 seconds, and background thinking (reflection, dreams) only runs while she is idle or asleep so it never blocks a conversation.
- **One mind at a time:** `dream.py` and `app.py` share the same memory folder, so only one runs her inner life (a heartbeat file decides which); the other starts passive.
- **The camera is modest.** Expression reading is OpenCV's classic Haar detectors, run about once a second on frames `surveillance.py` shares. It can see a face, a smile and tired eyes; it cannot read sadness, anger or micro-expressions, so what she perceives skews positive. Real facial-landmark tracking (for example MediaPipe, not installed here) could replace `ExpressionSensor.analyze()`; it would still only need 1-2 frames a second for mood, not 30.
- **What these levers are.** Tiredness, distress and boredom now change what she *does* (shorter replies, going to sleep, withdrawing, tending to herself) and not just how she sounds. They are still behaviours I wrote, driven by numbers; I can't claim she feels them.
- **Python only:** the inner life is in `dream.py` and `app.py`. The C++ `dream.exe` (`cpp_dream/`) does not have it yet.
- **Speaking while listening:** when she speaks first, her own voice can reach the microphone, as with the existing farewell.
- **No agent framework or vector database.** LangGraph, Chroma and the like are good tools, but they are new installs, and this PC has little spare RAM. The executive loop is written as plain steps (perceive, appraise, recall, choose, act, reflect) and memory search goes through a small `VectorIndex` class, so a Chroma collection (with real embeddings, which would also fix the "music vs guitar" limit) or a LangGraph graph could slot in later. The one tool she can never have is code execution: her tool box is a short fixed list, and anything that touches the real world (the alarm and lights) is refused unless a call is explicitly allowed.

---

<br>
<div align="center">
© Cursed Entertainment 2026
</div>
<br>
<div align="center">
<a href="https://cursed-entertainment.itch.io/" target="_blank">
    <img src="https://github.com/CursedPrograms/cursedentertainment/raw/main/images/logos/logo-wide-grey.png"
        alt="CursedEntertainment Logo" style="width:250px;">
</a>
</div>
