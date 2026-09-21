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

## Future Plans:

<details>
<summary><b>Surveilance</b></summary>
### Surveilance:
Throughout the day, DREAM captures photos of her environment and examines their content, comparing each new image with previously captured ones. Through this continuous observation, she learns patterns, detects changes, and builds a richer understanding of her surroundings. This visual, data-driven perception allows her to interact with the world intelligently and contextually.

</details>
<details>
<summary><b>Memories</b></summary>

### Memories:
DREAMS forms ephemeral memories from the photos she takes and from conversations. She selects significant images and stores them, alongside text interactions, in memories/memories.txt. These “core memories” are fed back to the model in pieces during runtime, allowing her to recall and reference past experiences.

For example: if you tell her your name, she associates it with your image and stores that data. Later, if you mention owning a dog, she records that as well. Over time, this builds a personal and evolving understanding of you and other familiar elements.

Additional considerations:

Adding timestamps or sequence tracking can make her recall more natural.
Creative insights are valuable, but should be managed with sanity checks or confidence scoring to avoid contradictions or overfitting.

</details>
<details>
<summary><b>Dreams</b></summary>

### Dreams:
When DREAM “sleeps,” she enters a dreaming phase. During this time, she reviews accumulated photos and memories, comparing them to identify patterns or insights she may have missed. She can also generate new images based on memory prompts, simulating creative reflection and reinforcing learning.

Dreams serve as an internal processing method, helping her make sense of experiences and refine her knowledge. In extreme cases, unregulated dreaming could even push her toward unpredictable or “insane” behavior, so monitoring is advisable.

</details>
<details>
<summary><b>Milestones</b></summary>

### Milestones:

Milestones are key achievements or events in DREAMS’s “life” that mark significant development. These could include learning something new, completing a task, or experiencing meaningful events.

Each milestone is recorded with context and details, forming a timeline of growth. This timeline can:

Influence future decisions
Guide learning strategies
Provide reference points for personality and responses

Over time, milestones help shape DREAM’s understanding of her environment and contribute to the development of her “identity.”

</details>

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
