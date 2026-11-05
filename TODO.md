# TODO

## Next session (left 2026-11-04)

- [ ] **Commit and push DREAM.** Nothing from 2026-11-03/04 is committed: stop-listening-on-silence
      (dream.js, scripts/dream.py, cpp_dream/audio.cpp), the three device settings (`Device`,
      `ImageDevice`, `PiperDevice` via scripts/dream_device.py, hints in options_hints.json),
      dream_generator.py (the Image-Generator repo folded in), the DeepDream fixes (nightmare preset,
      octaves too small), MuseTalk's model paths made absolute, run.bat's TensorFlow repair, the
      tf-keras pin, the deleted root setup.bat, `ModelBackup`, README / index.html, new demo_images.
      DREAM_XXX's index.html (gallery + DREAM updates) isn't committed either.
- [ ] **Build the C++ dream.exe** (`cpp_dream\setup.bat`, then `build.bat` in a new terminal): the
      silence detection, `Device` and the Ollama change in cpp_dream/ were only syntax-checked by eye
      (no g++ / ninja on PATH that night).
- [x] **SD-Turbo for her dream visions.** Done 2026-11-04 02:25: fetched into `F:\AIModels\sd-turbo`
      (the backup; Hugging Face's Xet CDN hung, so the workshop's 8-connection downloader did it) and
      copied to `models\sd-turbo`. Tested a full dream on the CPU (ImageDevice cpu): latent walk +
      DeepDream + SD-Turbo vision + 8 re-imagined frames in 195 s, `engine: latent+tensorflow+imagegen`;
      the spiral turns into the flooded-library vision like demo_images/dream_walk_into_vision.jpg.
- [ ] **Find the photo a robot took of the user** that they think was pushed. Searched every fleet
      repo's full history (local and GitHub) and everything under D:\cc: only artwork turned up.
      DREAM's photo folders are gitignored; where KIDA00/01, NINA, TINA, NORA save camera pictures
      wasn't found. Ask which robot / when. Removing it from GitHub means rewriting history: ask first.
- [ ] **When the P100 (16 GB) arrives** (planned for the second x16 slot, PCIe 2.0 x4 on the B450):
      add it as GPU 1 to the README's "Her body" table (specs only, no brands); test it in the
      return window (`nvidia-smi -q`: 16 GB, ECC errors, retired pages; 30-60 min load, temps);
      cap it with `nvidia-smi -pl` on the 550 W PSU (or a 750 W+ PSU); give dream_device a GPU
      number so the LLM + Whisper go on the P100 and the 2060 keeps lip-sync; try qwen3:14b.

## Lip-sync: faster and better than MuseTalk

Measured 2026-11-04 (RTX 2060, 6 GB): MuseTalk took ~45 s for 6.7 s of speech (3.5 fps) and
peaked at 6.7 GB, so it spills out of VRAM; the user thinks it looks bad too. `LipsyncEnabled`
stays off. Only scripts/dream.py lip-syncs: app.py / dream.js (what builds/dream.exe shows)
and cpp_dream have no lip-sync at all, so whatever wins has to go into the web page too.

MuseTalk claims 30+ fps on a V100 and was tested on a 4 GB RTX 3050 Ti, so part of the slowness
is ours: musetalk_talk.mp4 is 1920x1080 and every frame is composited at full HD on the CPU.

Try, cheapest first:
- [ ] **Wav2Lip** (already in `Wav2Lip/`; demo_images/lipsync_wav2lip.gif) on a 720p copy of the
      talking video, face boxes cached once. Time it on the same 6.7 s line. Blurry mouth (96x96),
      research-only licence. ONNX/OpenVINO ports run on the CPU (weaker sync).
- [ ] **MuseTalk on the 720p video**, to see how much was the resolution.
- [ ] **Ultralight-Digital-Human** (or its successor FeatherTalk): real time, made for phones, trained
      on a few minutes of video of her face (her talking videos). Most likely to be fast *and* look right.
- Not for her, from the search: Ditto (antgroup; its TensorRT path needs Ampere+, the 2060 is
  Turing), LiteAvatar (CPU real time but its own 2D avatar format), LatentSync 1.6 (best quality,
  too slow live). LiveTalking (github.com/lipku/LiveTalking) is a streaming framework that runs
  Wav2Lip / MuseTalk / Ultralight and could show how to wire one in.
- Cheaper fallbacks: lip-sync only her fixed lines once and cache them (dream.py's lipsync cache),
  or move the mouth from the voice's loudness.

## Smarter brain: move off phi3:mini

DREAM still talks with `phi3:mini` (`LlmModel` in `config.json`), the weakest part of her
stack. Plan:

- [x] **Switch her voice model to `qwen3:4b-instruct-2507`** (`ollama pull qwen3:4b-instruct-2507-q4_K_M`,
      ~2.5 GB): the same Qwen3 4B as TINA, minus the always-on thinking. Smarter than phi3
      (reasoning, maths, instructions), about as fast, never reasons out loud, and works with
      her existing prompts as they are, so only `LlmModel` changes.
      Done 2026-10-14 (`LlmModel`, the defaults in dream_models.py / dream_mind/llm.py, the options list).
      Tested on her real prompts: chat replies in ~4 s (phi3: 9-13 s, and it kept writing extra
      "Human: / You:" turns), in character, never thinking out loud; dreams and the reflection JSON
      come out right. It likes `*emphasis*` and emoji, which Piper reads out ("smirking face"), so
      `ask_llm` strips them (SPOKEN_JUNK in app.py and scripts/dream.py).
- [x] **Hand TINA's kind of work to TINA.** For multi-step tasks and tools, DREAM sends the task
      to TINA (`http://127.0.0.1:5013/`) instead of copying her code, and **speaks only TINA's
      final answer**.
      Done 2026-10-14: scripts/dream_fleet.py. "Hey TINA, ...", "ask TINA to ...", code / repo jobs and
      "why isn't X working" go to TINA's `POST /task`; DREAM follows `GET /task/<id>` and speaks only
      `result.answer` (no code blocks or markdown, cut at ~600 characters). TINA's questions are passed
      on for a spoken yes / no. `HandToTina` in config.json turns the automatic part off.
- [x] **"Show thoughts" toggle in the app**, off by default: TINA's reasoning (and anything else
      DREAM thinks) shown on screen when switched on. **Reasoning never goes to Piper.**
      Done 2026-10-14: `T` in the desktop app (a panel bottom left), 💭 on the browser page
      (dream.html, remembered per browser), `ShowThoughts` in config.json for the start. Shows TINA's
      reasoning, her tool calls and what RIFT / the robots answered.

### Why not TINA's `qwen3:4b` for DREAM's voice too

Tested 2026-10-13 on the RTX 2060 (Ollama):

- `qwen3:4b` is Qwen's thinking-only 2507 build (`thinking` capability, on by default).
  `"think": false` does **not** stop it: it reasons in the reply text instead
  (*"Hmm, the user is asking me…"*), which Piper would read out.
- `"think": true` on `/api/chat` with proper system/user messages separates the two cleanly
  (reasoning in `message.thinking`, answer in `message.content`), about 3 s per short reply.
- But with DREAM's one-string `System: … Human: … You:` prompt on `/api/generate` it over-thinks:
  1,174 tokens of reasoning, **no answer**, 14.5 s. Her dreaming and reflection use that style
  too, so the thinking model would need all of `scripts/dream_mind/llm.py` and its callers
  rewritten for chat messages plus a separate thinking budget. The instruct model avoids all of it.

## dream.exe (C++) catches up

`cpp_dream` (`build\dream.exe`) is a separate program, and it's behind `scripts/dream.py`.
(`builds\dream.exe` isn't: it shows her hosted page, so it has everything the page has.)

- [ ] **Waking like a person** (`InstantWake`): port `scripts/dream_wake.py` (1-10 calls from her
      sleepiness, sleepy mumbles with the real time) into brain.cpp's sleep/wake.
- [ ] **Add-ons**: load `addons/<name>/` (the `enabled` switch, the clips in place of hers, requests
      with a line and a clip). The add-ons are Python (`scripts/dream_addons.py`), so this needs
      them to describe their clips and requests in their `addon.json` as well.
- [ ] **Just her face**: Space shows or hides a status and caption overlay, like dream.py and the page.
      It has no overlay yet; T for thoughts needs the thoughts first.
- [ ] The rest of dream.py since it was written: the fleet (TINA, RIFT), her mind and sight, faces.
