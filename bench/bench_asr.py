"""Benchmark Armenian ASR candidates on CPU: WER, CER, latency, real-time factor.

  python bench_asr.py --models fastconformer-int8 whisper-small --n 40
"""
import argparse, time, os
import jiwer, soundfile as sf
from common import load_fleurs, norm_hy, pct

def make_fastconformer(quant):
    import onnx_asr
    from huggingface_hub import snapshot_download
    path = snapshot_download("Vana-Labs/stt-fastconformer-armenian",
        allow_patterns=["*.json", "*.txt", "nemo80.onnx"] +
        (["*.int8.onnx"] if quant else ["encoder-model.onnx", "decoder_joint-model.onnx"]))
    m = onnx_asr.load_model("nemo-conformer-rnnt", path, quantization="int8" if quant else None)
    def run(wav):                      # FLEURS wavs are float32 - decode ourselves
        x, sr = sf.read(wav, dtype="float32")
        return m.recognize(x, sample_rate=sr)
    return run

def make_fw(name, threads):
    from faster_whisper import WhisperModel
    m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
    def run(wav):
        x, _ = sf.read(wav, dtype="float32")   # FLEURS is already 16 kHz mono
        segs, _ = m.transcribe(x, language="hy", beam_size=1, vad_filter=False,
                               condition_on_previous_text=False)
        return " ".join(s.text for s in segs)
    return run

MODELS = {
    "fastconformer-int8": lambda t: make_fastconformer(True),
    "fastconformer-fp32": lambda t: make_fastconformer(False),
    "whisper-small":      lambda t: make_fw("small", t),
    "whisper-medium":     lambda t: make_fw("medium", t),
    "whisper-large-v3-turbo": lambda t: make_fw("large-v3-turbo", t),
    # Armenian fine-tunes (convert first: see README_bench.md) - pass a CT2 dir via path:<dir>
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["fastconformer-int8"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    data = load_fleurs(a.n)
    print(f"{len(data)} clips, {a.threads} threads\n")
    for name in a.models:
        if name.startswith("path:"):
            make = lambda t, p=name[5:]: make_fw(p, t)
        else:
            make = MODELS[name]
        t0 = time.perf_counter(); run = make(a.threads); load = time.perf_counter() - t0
        run(data[0]["wav"])                      # warm-up
        refs, hyps, lat, dur = [], [], [], []
        for d in data:
            info = sf.info(d["wav"]); dur.append(info.duration)
            t = time.perf_counter(); h = run(d["wav"]); lat.append(time.perf_counter() - t)
            refs.append(norm_hy(d["hy"])); hyps.append(norm_hy(h))
        rtf = sum(lat) / sum(dur)
        print(f"== {name}")
        print(f"   load {load:.1f}s | WER {jiwer.wer(refs, hyps)*100:.1f}% | CER {jiwer.cer(refs, hyps)*100:.1f}%")
        print(f"   latency p50 {pct(lat,.5):.2f}s p95 {pct(lat,.95):.2f}s | RTF {rtf:.2f} "
              f"(avg clip {sum(dur)/len(dur):.1f}s)")
        print(f"   sample REF: {refs[1]}\n   sample HYP: {hyps[1]}\n")

if __name__ == "__main__":
    main()
