"""Armenian speech recognition: NVIDIA FastConformer (hybrid transducer) as int8 ONNX, via onnx-asr."""
import re
import numpy as np

SR = 16000

REPO = "Vana-Labs/stt-fastconformer-armenian"


def fix_armenian(text: str) -> str:
    """The model writes the ew ligature as two letters; use the standard single glyph, and it often starts
    a line with a dialogue dash (U+058A / -) which confuses the translator - drop those."""
    t = text.replace("եւ", "և").strip()
    return re.sub(r"^[\s\u058a\-\u2013\u2014\u00b7\"'«»]+", "", t).strip()


class Recognizer:
    def __init__(self, threads: int = 2):
        import onnx_asr, onnxruntime as ort
        from huggingface_hub import snapshot_download
        path = snapshot_download(REPO, allow_patterns=["*.json", "*.txt", "nemo80.onnx", "*.int8.onnx"])
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.m = onnx_asr.load_model("nemo-conformer-rnnt", path, quantization="int8",
                                     sess_options=so)

    @staticmethod
    def normalise(x: np.ndarray, target_rms: float = 0.1, max_gain: float = 30.0) -> np.ndarray:
        """Bring quiet microphones up to a standard level (cut WER 6.8 -> 5.7 % on -25 dB test audio)."""
        rms = float(np.sqrt(np.mean(x ** 2))) + 1e-6
        return np.clip(x * min(target_rms / rms, max_gain), -1, 1).astype(np.float32)

    @staticmethod
    def pad(x: np.ndarray, seconds: float = 0.4) -> np.ndarray:
        """VAD cuts start right at the first sound; the model garbles abrupt onsets (e.g. 'Պատմութիւնը' ->
        'Մմութիւնն'). Surround the phrase with low noise that matches its own noise floor (digital silence
        made noisy audio worse: WER 18 -> 23 %; matched noise: 18 -> 18, clean 7.9 -> 5.8 %)."""
        n = int(SR * seconds)
        frames = x[: len(x) // 320 * 320].reshape(-1, 320)
        floor = float(np.sqrt(np.percentile(np.mean(frames ** 2, axis=1), 10))) if len(frames) else 0.0
        edge = lambda: (np.random.randn(n) * floor).astype(np.float32)
        return np.concatenate([edge(), x, edge()])

    def __call__(self, samples: np.ndarray) -> str:
        samples = self.pad(self.normalise(samples))
        return fix_armenian(self.m.recognize(samples, sample_rate=16000))
