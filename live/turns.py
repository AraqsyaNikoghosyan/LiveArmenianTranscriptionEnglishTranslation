"""Speaker-turn splitting inside one VAD phrase.

On YouTube / TV talk, people answer each other without a pause, so one VAD phrase can contain two
speakers. A neural segmentation model (pyannote-3.0, ONNX, ~7 MB) finds where the voice changes;
we cut there and let the global speaker tracker name each piece.
"""
from __future__ import annotations
import numpy as np
import sherpa_onnx as so

SR = 16000


class TurnSplitter:
    def __init__(self, segmentation_model: str, embedding_model: str, threshold: float = 0.7,
                 min_piece: float = 0.8, threads: int = 1):
        cfg = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation_model),
                num_threads=threads),
            embedding=so.SpeakerEmbeddingExtractorConfig(model=embedding_model, num_threads=threads),
            clustering=so.FastClusteringConfig(num_clusters=-1, threshold=threshold),
            min_duration_on=0.3, min_duration_off=0.3)
        self.sd = so.OfflineSpeakerDiarization(cfg)
        self.min_piece = int(min_piece * SR)

    def split(self, x: np.ndarray) -> list[tuple[int, int]]:
        """Return [(start, end)] sample ranges covering x. One range = no speaker change found."""
        n = len(x)
        segs = [(s.start, s.end, s.speaker) for s in self.sd.process(x).sort_by_start_time()]
        cuts = []
        for (s0, e0, k0), (s1, e1, k1) in zip(segs, segs[1:]):
            if k1 != k0:
                cuts.append(int((min(e0, s1) + s1) / 2 * SR) if e0 > s1 else int((e0 + s1) / 2 * SR))
        bounds = [0] + sorted(c for c in cuts if 0 < c < n) + [n]
        # drop cuts that would leave a piece shorter than min_piece (merge into the shorter neighbour)
        while len(bounds) > 2:
            lens = [b - a for a, b in zip(bounds, bounds[1:])]
            i = int(np.argmin(lens))
            if lens[i] >= self.min_piece: break
            left = lens[i - 1] if i > 0 else 1 << 60
            right = lens[i + 1] if i < len(lens) - 1 else 1 << 60
            drop = i if (i > 0 and left <= right) else i + 1
            del bounds[drop]
        return list(zip(bounds, bounds[1:]))
