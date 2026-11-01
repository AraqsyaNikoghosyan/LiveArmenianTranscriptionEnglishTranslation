"""Wires everything together.

 source(s) -> VAD (one per source) -> phrase queue -> [turn split + speaker label + ASR] -> translation pool -> ordered print

 * One VAD per audio source, so mic and system audio never mix in a phrase.
 * ASR is a single worker (the CPU is the bottleneck, and order matters for speaker tracking).
 * Translation runs in a small pool so the next phrase can already be recognised meanwhile.
 * Timing uses the wall clock of when audio really arrived (WASAPI loopback sends nothing during silence,
   so counting samples would drift away from real time).
"""
from __future__ import annotations
import bisect, queue, re, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vad import PhraseDetector, Segment
from .speakers import OnlineSpeakerTracker, split_on_speaker_change

SR = 16000


@dataclass
class Settings:
    vad_model: str
    speaker_model: str
    translate_dir: str
    segmentation_model: str = ""
    spk_threshold: float = 0.45
    min_silence: float = 0.45
    max_speech: float = 8.0
    split_thr: float = 0.30        # speaker-change score below which a phrase may be split (0 = never split)
    beam: int = 1
    show_latency: bool = False
    split_turns: bool = True
    splitter: str = "pyannote"    # "pyannote" (neural segmentation) or "embed" (window embeddings, no extra model)
    dump_dir: str = ""
    threads: int = 4


class Pipeline:
    def __init__(self, sources: dict, cfg: Settings):
        from .asr import Recognizer
        from .translate import Translator
        from .speakers import Embedder
        self.sources, self.cfg = sources, cfg
        t = time.time()
        print("Loading models ...", flush=True)
        self.asr = Recognizer(threads=max(1, cfg.threads // 2))
        self.mt = Translator(cfg.translate_dir, threads=max(1, cfg.threads // 2), workers=2, beam=cfg.beam)
        self.embed = Embedder(cfg.speaker_model, threads=max(1, cfg.threads // 2)) if "system" in sources else None
        self.splitter = None
        if "system" in sources and cfg.split_turns and Path(cfg.segmentation_model).exists():
            from .turns import TurnSplitter
            self.splitter = TurnSplitter(cfg.segmentation_model, cfg.speaker_model)
        elif "system" in sources and cfg.split_turns:
            print("  (turn splitting disabled: run setup_models.py to download the segmentation model)")
        self.tracker = OnlineSpeakerTracker(threshold=cfg.spk_threshold)
        z = (np.random.randn(SR * 2) * 0.01).astype("float32")      # warm-up so the first phrase isn't slow
        self.asr(z); self.mt("Բարև")
        if self.embed: self.embed(z)
        if self.splitter: self.splitter.split(z)
        print(f"Ready in {time.time() - t:.1f}s. Listening ... (Ctrl+C to stop)\n", flush=True)

        self.segq: "queue.Queue" = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.print_lock = threading.Lock()
        self.results: dict[int, str] = {}
        self.next_print = 0
        self.seq = 0
        self.t_zero = time.time()
        self.stop_ev = threading.Event()
        if cfg.dump_dir: Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)

    # ---- stage 1: audio -> phrases
    def _vad_loop(self, name: str, src):
        vad = PhraseDetector(self.cfg.vad_model, min_silence=self.cfg.min_silence, max_speech=self.cfg.max_speech)
        pads = getattr(src, "pads_silence", False)
        last = time.time()
        while not self.stop_ev.is_set():
            try: item = src.q.get(timeout=0.1)
            except queue.Empty:
                now = time.time()
                if pads and now - last >= 0.1:        # source sends nothing in silence (WASAPI loopback): feed real-time silence
                    for seg in vad.feed(np.zeros(int(SR * (now - last)), np.float32), name, now): self.segq.put(seg)
                    last = now
                continue
            if item is None:
                for seg in vad.flush(name, time.time()): self.segq.put(seg)
                break
            chunk, wall = item
            for seg in vad.feed(chunk, name, wall): self.segq.put(seg)
            last = time.time()
        self.segq.put(("done", name))

    # ---- stage 2: ASR + speaker label (sequential)
    def _asr_loop(self):
        open_sources = len(self.sources)
        while open_sources:
            item = self.segq.get()
            if isinstance(item, tuple):
                open_sources -= 1; continue
            seg: Segment = item
            whole = [(0, len(seg.samples))]
            if seg.source == "mic" or not self.cfg.split_turns:
                parts = whole
            elif self.splitter is not None and self.cfg.splitter == "pyannote":   # system audio: people may answer without a pause
                parts = self.splitter.split(seg.samples)
            elif self.cfg.split_thr > 0:
                parts = split_on_speaker_change(self.embed, self.tracker, seg.samples, change_thr=self.cfg.split_thr)
            else:
                parts = whole
            for a, b in parts:
                x = seg.samples[a:b]
                if len(x) < 8000: continue                               # < 0.5 s: noise
                hy = self.asr(x)
                if sum(c.isalpha() for c in hy) < 2: continue            # silence / noise / lone sound
                if seg.source == "mic":
                    label = "Me"
                else:
                    label = f"Speaker {self.tracker.assign(self.embed(x[: 16000 * 10]), len(x) / 16000) + 1}"
                i = self.seq; self.seq += 1
                if self.cfg.dump_dir:
                    import soundfile as sf
                    sf.write(Path(self.cfg.dump_dir) / f"{i:04d}_{label.replace(' ', '')}.wav", x, SR)
                    (Path(self.cfg.dump_dir) / f"{i:04d}_{label.replace(' ', '')}.txt").write_text(hy, encoding="utf-8")
                start = seg.start - seg.pad_pre + a / 16000
                self.pool.submit(self._translate, i, start, seg.end_wall, label, hy)
        self.pool.shutdown(wait=True)

    # ---- stage 3: translation + ordered output
    def _translate(self, i: int, start: float, end_wall: float, label: str, hy: str):
        try: en = self.mt(hy)
        except Exception as e: en = f"[translation error: {e}]"
        done = time.time()
        t = max(0, int(start - self.t_zero)); stamp = f"[{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}]"
        lat = f"   (latency {done - end_wall:.1f}s)" if self.cfg.show_latency else ""
        block = f"{stamp} {label}{lat}\nHY: {hy}\nEN: {en}\n"
        with self.print_lock:
            self.results[i] = block
            while self.next_print in self.results:
                print(self.results.pop(self.next_print), flush=True); self.next_print += 1

    def run(self):
        self.t_zero = time.time()
        for name, src in self.sources.items(): src.start()
        threads = [threading.Thread(target=self._vad_loop, args=(n, s), daemon=True) for n, s in self.sources.items()]
        for t in threads: t.start()
        asr = threading.Thread(target=self._asr_loop, daemon=True); asr.start()
        try:
            while asr.is_alive(): asr.join(0.3)
        except KeyboardInterrupt:
            print("\nStopping ...")
            self.stop_ev.set()
            for s in self.sources.values(): s.stop()
