"""Speaker identification for the system-audio channel (local, CPU, pip-only).

Embedder          : sherpa-onnx speaker-embedding model (ONNX, no torch needed)
OnlineSpeakerTracker : assigns each utterance to "Speaker N", creating new speakers on the fly.
"""
from __future__ import annotations
import numpy as np


class Embedder:
    def __init__(self, model_path: str, threads: int = 2):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path, num_threads=threads, provider="cpu")
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def __call__(self, samples: np.ndarray, sr: int = 16000) -> np.ndarray:
        s = self.ex.create_stream()
        s.accept_waveform(sr, np.ascontiguousarray(samples, dtype=np.float32))
        s.input_finished()
        v = np.asarray(self.ex.compute(s), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


class OnlineSpeakerTracker:
    """Greedy online clustering on cosine similarity.

    - sim to best centroid >= threshold  -> that speaker; centroid is updated
    - otherwise                           -> new speaker (up to max_speakers, then best match)
    - utterances shorter than min_dur     -> embedding is unreliable: reuse best match
                                             (or last speaker) and do NOT update centroids
    - 'margin' band just above the threshold: assign but don't update (avoids centroid drift)
    """

    def __init__(self, threshold=0.5, min_dur=1.0, max_speakers=8, margin=0.05):
        self.thr, self.min_dur, self.max_spk, self.margin = threshold, min_dur, max_speakers, margin
        self.sums: list[np.ndarray] = []
        self.last = None

    def _sims(self, e):
        return np.array([float(e @ (s / np.linalg.norm(s))) for s in self.sums])

    def assign(self, emb: np.ndarray, dur: float) -> int:
        """Returns a 0-based speaker index."""
        if not self.sums:
            if dur < self.min_dur:
                return 0                       # placeholder; first reliable segment will create spk 0
            self.sums.append(emb.copy()); self.last = 0
            return 0
        sims = self._sims(emb)
        best = int(sims.argmax())
        if dur < self.min_dur:                 # too short to trust
            self.last = best if sims[best] >= self.thr else (self.last or 0)
            return self.last
        if sims[best] >= self.thr:
            if sims[best] >= self.thr + self.margin:
                self.sums[best] += emb
            self.last = best
        elif len(self.sums) < self.max_spk:
            self.sums.append(emb.copy()); self.last = len(self.sums) - 1
        else:
            self.last = best
        return self.last
