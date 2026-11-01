# Live Armenian → English call translator

A Python program that listens to a call and prints, line by line, who said what in Armenian and English.
Everything runs **locally on the CPU** (no cloud speech or translation APIs, no GPU).

```
[00:01:12] Speaker 1   (latency 2.6s)
HY: Բարև, ինչպե՞ս ես։
EN: Hello, how are you?

[00:01:15] Me   (latency 0.9s)
HY: Լավ եմ, շնորհակալություն։
EN: I'm fine, thank you.
```

* **Microphone** → labelled `Me`.
* **System audio** (the other side of a call, a YouTube video, ...) → `Speaker 1`, `Speaker 2`, ... The same
  voice keeps its label for the whole session.
* Developed and tested on: Windows 11, Python 3.12, 4 logical cores, no GPU.

---

## 1. Setup and run

### Requirements
* Windows 10/11 (audio capture uses WASAPI, see "Audio device setup"), Python 3.12.
* About 6 GB free disk space (models are downloaded on first use; they are **not** in the repository).
* `ffmpeg` is only needed for the optional benchmark / simulation tools (`winget install Gyan.FFmpeg`).

### Install
```
python -m venv .venv
.venv\Scripts\activate            (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu    # only needed once, for step below
python setup_models.py
```
`setup_models.py` downloads the voice-activity, speaker and segmentation models, fetches the Armenian speech
model, and converts the translation model (NLLB-200) to a fast int8 format. It takes a few minutes the first time.
After it has finished, `torch` is not needed any more.

### Run
```
python -m live --list-devices        # shows microphones and loopback ("system audio") devices with ids
python -m live                       # default microphone + loopback of the default speakers
python -m live --mic 3 --loopback 17 # pick devices explicitly (ids from --list-devices)
python -m live --show-latency        # also print end-of-speech -> English delay for every line
```
Stop with `Ctrl+C`.

| Option | Meaning |
|---|---|
| `--no-mic` / `--no-system` | listen to only one of the two sources |
| `--spk-threshold 0.45` | speaker similarity threshold. Lower → fewer speakers, higher → more speakers |
| `--min-silence 0.45` | pause (seconds) that ends a phrase. Higher → fewer, longer phrases but more delay |
| `--max-speech 8` | phrases longer than this are cut (bounds the latency) |
| `--splitter pyannote\|embed`, `--no-split` | how a phrase containing two speakers is split (see Architecture) |
| `--beam 1` | translation beam size (1 = fastest) |
| `--threads N` | CPU threads to use (default: all cores) |
| `--dump DIR` | save every recognised phrase as `.wav` + `.txt` to debug recognition |
| `--sim MIC.wav SYSTEM.wav` | run without audio devices: play two wav files as the two sources |
| `--sim-loopback` | with `--sim`: drop silent chunks like WASAPI loopback does |

### Audio device setup (Windows)
No virtual cable is needed. Windows can record "what you hear" from any output device (WASAPI loopback), and
the program does that through [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch).

1. Play the call / video through your normal speakers or headphones (set them as the Windows default output).
2. `python -m live --list-devices`: the *loopback* list contains one entry per output device. The one marked
   `<- default output` is used automatically. Use `--loopback ID` for another one.
3. **Use headphones.** Echo cancellation is out of scope: with speakers, the microphone hears the call and
   it will show up as `Me` lines.
4. Telegram / Zoom / Meet: whatever you hear through the default output is captured.
   macOS (BlackHole + Multi-Output Device) is **not supported by this code**; only Windows was implemented and tested.

### Testing without audio devices
```
python speaker/fetch_speaker_data.py                                     # Common Voice Armenian test split
python tools/make_sim_call.py --out sim                                  # builds sim_mic.wav / sim_system.wav + ground truth
python tools/make_sim_call.py --out hard --gap 0 0.05 --tight --order A,B,A,A,B,B   # speakers answering without a pause
python -m live --sim sim_mic.wav sim_system.wav --show-latency
```
This builds a fake call from real Armenian recordings of different people, with a printed ground truth.

---

## 2. Architecture

```
 mic (WASAPI)  ─┐                                   ┌─ ASR (FastConformer int8) ─┐
                ├─► resample 16 kHz mono ─► VAD ─► phrases ─► [speaker-turn split] ─► speaker label ─► translate ─► ordered print
 loopback      ─┘   (one VAD per source)  (Silero)   queue      (system audio only)   (CAM++ + online     (NLLB-600M    (HY + EN)
                                                                                         clustering)       int8, 2 workers)
```

| File | Responsibility |
|---|---|
| `live/audio.py` | device listing, WASAPI capture (mic + loopback), file simulator. Delivers 16 kHz mono chunks with a wall-clock stamp |
| `live/vad.py` | Silero VAD (via sherpa-onnx) cuts the stream into phrases and adds a little padding around them |
| `live/asr.py` | Armenian speech recognition (NVIDIA FastConformer, int8 ONNX) + text cleanup |
| `live/speakers.py` | speaker embedding (3D-Speaker CAM++), online speaker tracker, embedding-based turn splitter |
| `live/turns.py` | neural speaker-turn splitter (pyannote segmentation 3.0, ONNX) |
| `live/translate.py`, `live/lexicon.py` | NLLB-200 translation (CTranslate2) + special handling of very short utterances |
| `live/pipeline.py` | threads and queues that connect everything; ordered output |
| `bench/`, `speaker/` | the benchmark scripts behind the model choices below |
| `tools/make_sim_call.py` | builds a simulated call with ground truth |

**Threads.** Each source has its own capture and VAD thread, so mic and system audio never get mixed in one
phrase. A single worker does speech recognition and speaker labelling in order (the speaker tracker is stateful and
the CPU is the bottleneck anyway). Translation runs in a pool of two so the next phrase can be recognised
while the previous one is still being translated. Output is printed in order.

**Speaker identification.** Everything from the microphone is `Me`. For system audio each phrase is turned into a
voice embedding (CAM++). `OnlineSpeakerTracker` compares it with the running average of every known speaker by
cosine similarity: above the threshold (0.45) it is that speaker, otherwise a new speaker is created. Phrases
shorter than 1 s are too short for a reliable embedding, so they reuse the closest speaker without changing it.
Only confident matches update a speaker's average, which avoids drift.

**Speaker changes inside a phrase.** In real conversations people answer without a pause, and then the VAD returns
one phrase with two voices. For system audio the phrase is therefore split at speaker changes
(pyannote-segmentation-3.0 by default; a fallback that compares embeddings of sliding windows is selectable with
`--splitter embed`), and every part is recognised and labelled on its own.

**Things that were needed to make it work on a real machine (found while testing):**
* *Padding.* A VAD cuts exactly at the speech edge, which clips soft consonants ("Բարև" → "Պարև"). Adding 0.25 s before and
  0.15 s after each phrase lowered the word error rate from 9.0 % to 6.7 % on 60 Common Voice clips.
* *Silence on loopback.* WASAPI loopback delivers **no data while nothing is playing**. Counting samples then drifts
  away from real time (we first saw bogus "20 s latency" and wrong timestamps, and the last phrase before a pause
  would not be finished). The pipeline now feeds real-time silence into the VAD when a loopback source is quiet and
  derives all times from the wall clock.
* *Very short utterances.* NLLB invents sentences for one-to-three-word inputs ("Բարև" → "I'm going to be a little
  late", "Ո՞նց ես" → "Who are you?"). Up to four words are therefore translated inside a carrier sentence
  (`Նա ասաց՝ «…»` with the English forced to start with `He said, "`), and a small phrase lexicon (`lexicon.py`) covers
  greetings and one-word answers. Longer sentences are translated directly.
* *Text cleanup.* The speech model writes the "ew" ligature as two letters and often starts a line with a dialogue
  dash; both are normalised before translation.

---

## 3. Models: what was chosen, what was considered, why

All numbers were measured with the scripts in `bench/` and `speaker/` (on the target machine, an Intel 4-core CPU,
unless stated otherwise). Armenian is low-resource, so quality differs a lot between models and the choice was
made by measuring, not by reputation.

### Speech recognition: `Vana-Labs/stt-fastconformer-armenian` (int8 ONNX export of `nvidia/stt_hy_fastconformer_hybrid_large_pc`)
* 115 M parameters, trained on ~296 h of Armenian (Common Voice 17, FLEURS, audiobooks and other corpora);
  outputs punctuation and capitalisation; licence CC-BY-4.0 (commercial use allowed).
* NVIDIA reports 9.9 % WER on Common Voice 17 and 12.3 % on FLEURS test. On the FLEURS dev clips used by my benchmark it
  got **2.4 % WER, p95 latency 0.45 s per ~11 s clip (real-time factor 0.02)**. FLEURS was part of its training data, so
  this number is optimistic: treat the 10-12 % reported by NVIDIA as the realistic read-speech figure.
* The ONNX export runs with `onnx-asr` and `onnxruntime`: no NeMo, no PyTorch, no GPU, installs with pip on Windows.

| Alternative | Why not |
|---|---|
| Whisper `small` (multilingual, faster-whisper int8) | **95 % WER** on the same clips, and 3.2× slower than real time (measured on a 2-core machine). Whisper's Armenian is essentially unusable at this size |
| Whisper `medium` / `large-v3-turbo` multilingual | not measured: even larger and slower, and the small model was already far off |
| Armenian Whisper fine-tunes (`Chillarmo/whisper-large-v3-turbo-armenian`: 15.3 % WER self-reported on Common Voice 20; `whisper-small-armenian-v2`: 24 %; `arampacha/whisper-large-hy`: 22 %) | not benchmarked here. The large-v3-turbo one has 0.8 B parameters (its card asks for 8 GB GPU memory) and would not fit the 5 s budget on 4 CPU cores; the small ones have a clearly worse error rate than FastConformer |
| `milenabazoyan/fastconformer-hybrid-arm-asr` (8.5 % WER on Common Voice 20) | licence CC-BY-NC (non-commercial), needs NeMo (Linux preferred); not tried |

### Translation: `facebook/nllb-200-distilled-600M` converted to CTranslate2 int8
Measured on 40 FLEURS sentences (BLEU against the English reference, case-insensitive):

| Model | BLEU (true Armenian text) | BLEU (text from the speech model) | Latency per sentence (p50 / p95) |
|---|---|---|---|
| **NLLB-200 distilled 600M, int8** | **30.2** | **28.9** | 1.5 s / 3.0 s (3.7 s with ASR text) |
| Helsinki-NLP/opus-mt-hy-en, int8 | 14.0 | - | 0.2 s / 1.4 s |

The Marian model is faster but its output is poor and it invents names ("Aristarchus" → "Manfred"). NLLB-1.3B is
supported by `bench/bench_mt.py` but was not measured: it would be about twice as slow and the 5 s budget is
already mostly spent by the 600M model on long sentences. An LLM-based translator was not tried (too slow on CPU
for this budget). Weaknesses of NLLB are listed under limitations.

### Speaker identification: 3D-Speaker CAM++ embeddings (sherpa-onnx) + online clustering
Simulated calls with 2, 3 and 4 speakers built from Common Voice Armenian (client-id = speaker), mean accuracy after
the best label matching, best threshold per model:

| Embedding model | Accuracy, clean | Accuracy, Opus 16 kbps (phone-call like) | Time per phrase |
|---|---|---|---|
| **3D-Speaker CAM++ (zh_en, "common advanced")** | **99 %** (thr 0.40) | 96 % (sandbox run) | ~45 ms |
| NeMo TitaNet-small | 98 % (thr 0.40) | 98 % | ~45 ms |
| 3D-Speaker ERes2Net base | 94 % | 91 % | 165 ms (sandbox) |
| WeSpeaker ResNet34 (CN-Celeb) | 92 % | 87 % | 120 ms (sandbox) |
| WeSpeaker ResNet34 (VoxCeleb) | 78 % | - | 120 ms (sandbox) |

The CAM++ and TitaNet rows and the 45 ms timings were measured on the target PC (Opus figure for CAM++ and all rows for the last three models: 2-core cloud machine, same data and scripts).
CAM++ and TitaNet are equally good; CAM++ is used with threshold 0.45 (accuracy is flat between 0.40 and 0.50).
`pyannote` pipelines (the usual choice) were not used because their models are gated and need a token / PyTorch.

### Other components
* **VAD:** Silero VAD through sherpa-onnx (small, CPU, no PyTorch).
* **Speaker-turn segmentation:** `pyannote-segmentation-3.0` ONNX export from sherpa-onnx (7 MB).
  On a hard test (ten turns with **no pause** between different speakers) splitting off gave 4 lines, all `Speaker 1`; splitting on
  gave labels that followed the real speakers (a few turns were cut into extra lines).

### Reproducing the benchmarks
```
pip install pyarrow jiwer sacrebleu faster-whisper psutil        (and ffmpeg on PATH)
cd bench   && python fetch_data.py && python bench_asr.py --models fastconformer-int8 whisper-small --n 40
             python bench_mt.py --models opus-mt nllb-600m --source gold --n 40     (--source asr: text from the speech model)
cd speaker && python fetch_speaker_data.py && python bench_speakers.py --models models/*.onnx [--degrade opus]
```
See `bench/README_bench.md` for caveats (FLEURS was in the speech model's training data; Common Voice speakers each
used their own microphone).

---

## 4. Results and honest status

| What | Result |
|---|---|
| Armenian recognition on clean read speech | 2.4 % WER (optimistic, see above) |
| Armenian recognition on a Armenian YouTube talk show (real TV audio) | transcript visibly follows the video's own transcript; errors on proper names and spoken-language words. No WER was computed |
| Microphone path on the target PC | 0.7-1.2 s from end of speech to English line |
| Simulated 90 s call, 3 speakers, 2-core cloud machine | all 12 turns labelled correctly; delay 1.1-3.0 s |
| Same, speakers answering without a pause | labels followed the real speakers; delay up to 5-7 s on the 2-core machine (continuous speech has no idle time) |

What I did **not** verify: the 5-second limit for *continuous system-audio speech on the 4-core target machine after the
last fixes*, speaker accuracy on real YouTube/Telegram audio with a threshold tuned for it, and a real call with
overlapping speech. Use `--show-latency` to check on your own setup.

## 5. Known limitations
* **Windows only.** No macOS / Linux capture code.
* **Echo.** Out of scope: with loudspeakers, the microphone transcribes the call as `Me`.
* **Short utterances are the weak point.** One- and two-word replies ("yes", "OK", names, interjections) get
  wrong text from the speech model (it was trained on read speech) and sometimes a wrong translation; the
  lexicon covers only common greetings.
* **Noise and microphone quality matter more than anything else.** With white noise 10 dB below the voice, the error rate went from 6.7 % to
  21.7 % (simulated). Gain normalisation did not help.
* **Speaker labels** were tuned on recordings where every person used their own microphone. Same-channel voices
  (YouTube, a shared room) and overlapping speech are harder; a speaker can be split in two or two similar voices
  merged. Maximum 8 speakers; labels are not kept between sessions (by design).
* **Phrases are cut at pauses of 0.45 s and at 8 s,** so a sentence can be translated in two halves. A longer
  `--min-silence` trades delay for fewer cuts.
* **Translation** is sentence by sentence without context; NLLB-600M mistranslates rare words, idioms and
  colloquial Armenian, and softens insults ("ապուշ" → "buddy"). Western Armenian orthography is not supported by the speech model.
* **Latency** depends on the phrase length: ~0.5 s of VAD wait + recognition (~0.3 s) + speaker work (~0.1-1 s) + translation (0.2-3.5 s).
  Long, uninterrupted speech on a slow CPU can exceed 5 s.

## 6. What I would do with more time
* Test and tune on a labelled set of real calls (YouTube + Telegram) instead of simulated ones; add automatic WER/latency/DER reports.
* Noise suppression in front of the speech model (e.g. DeepFilterNet / RNNoise) and optional echo cancellation.
* Fine-tune the speech model on conversational Armenian; add confidence scores and hide low-confidence fragments.
* Show the Armenian text immediately and add the English afterwards, and translate long phrases in pieces, to cut the perceived delay.
* Context-aware translation (previous sentences), a larger or fine-tuned MT model if a GPU is available.
* Real diarization with overlap handling, and speaker re-identification over a longer window.
* Tests and a configuration file; macOS/Linux capture back-ends.

## 7. Repository notes
* Model weights, recordings and API keys are never committed (`models/`, `*.wav`, datasets are in `.gitignore`).
* `chats/` contains the full, unedited exports of all AI-assistant sessions used for this project.
