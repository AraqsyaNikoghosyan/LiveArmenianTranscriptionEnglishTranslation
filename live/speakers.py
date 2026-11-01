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


# --------------------------------------------------------------------------- speaker changes inside a phrase
def _peek(tracker, emb):
    """Which known speaker does this embedding belong to? (None = nobody) - does not update the tracker."""
    if not tracker.sums: return None
    sims = tracker._sims(emb)
    b = int(sims.argmax())
    return b if sims[b] >= tracker.thr else None


def split_on_speaker_change(embed, tracker, x: np.ndarray, sr: int = 16000, win=1.0, hop=0.5,
                            change_thr=0.30, min_part=1.0, max_cuts=3):
    """Phrases are cut by silence, but in a lively conversation people answer without a pause, so one
    phrase can contain two speakers. Returns [(start, end), ...] sample ranges, one per speaker turn.

    1. embed 1 s windows every 0.5 s
    2. for each candidate boundary compare everything before it with the 1 s after it; the lowest score wins
    3. snap the cut to the quietest point nearby (people breathe between turns)
    4. only keep the cut if the tracker thinks the two halves are different speakers, so a false alarm
       inside one person's sentence costs nothing
    """
    cache = {}                                                    # window index -> embedding (windows sit on a fixed grid)
    def win_emb(j):
        if j not in cache:
            cache[j] = embed(x[int(j * hop * sr): int((j * hop + win) * sr)])
        return cache[j]

    def find(a, b):
        seg = x[a:b]
        if len(seg) / sr < 3.0: return None
        j0 = -(-a // int(hop * sr))                               # first grid window starting at/after a
        js = [j for j in range(j0, 10 ** 6) if int((j * hop + win) * sr) <= b]
        n = len(js)
        if n < 5: return None
        W = np.array([win_emb(j) for j in js])
        off = j0 * hop * sr - a                                   # grid offset inside seg (samples)
        best = None
        for k in range(2, n - 1):
            L, R = W[: k - 1].sum(0), W[k + 1: k + 3].sum(0)
            s = float(L @ R / (np.linalg.norm(L) * np.linalg.norm(R) + 1e-9))
            if best is None or s < best[1]: best = (k, s)
        k, score = best
        if score >= change_thr: return None
        c = int((k * hop + hop / 2) * sr + off)                   # centre of the gap between the two context blocks
        lo, hi = max(int(sr * 0.5), c - int(0.6 * sr)), min(len(seg) - int(sr * 0.5), c + int(0.6 * sr))
        if hi <= lo: return None
        fr = int(0.03 * sr)
        e = [np.sqrt(np.mean(seg[i:i + fr] ** 2)) for i in range(lo, hi, fr // 2)]
        cut = lo + int(np.argmin(np.convolve(e, np.ones(3) / 3, "same"))) * (fr // 2)
        if cut < min_part * sr or len(seg) - cut < min_part * sr: return None
        eL, eR = embed(seg[:cut][-sr * 4:]), embed(seg[cut:][: sr * 4])
        lL, lR = _peek(tracker, eL), _peek(tracker, eR)
        if lL is not None and lL == lR: return None             # same known speaker: false alarm
        if lL is None and lR is None and float(eL @ eR) > 0.4: return None
        if not tracker.sums and score > change_thr - 0.05: return None   # nothing to confirm with: be strict
        return a + cut

    def rec(a, b, budget):
        if budget[0] <= 0: return [(a, b)]
        c = find(a, b)
        if c is None: return [(a, b)]
        budget[0] -= 1
        return rec(a, c, budget) + rec(c, b, budget)

    return rec(0, len(x), [max_cuts])
