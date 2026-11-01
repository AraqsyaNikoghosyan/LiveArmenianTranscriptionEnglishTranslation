"""Benchmark Armenian->English MT on CPU: BLEU/chrF + latency.

  python bench_mt.py --models opus-mt nllb-600m --source gold
  python bench_mt.py --models opus-mt nllb-600m --source asr   # real pipeline: ASR text in
"""
import argparse, time, os, subprocess, sys
import sacrebleu
from pathlib import Path
from common import load_fleurs, pct

CT2 = Path(__file__).parent / "ct2_models"

def convert(hf_id, out):
    """One-time HF -> CTranslate2 int8 conversion (needs torch + transformers)."""
    if not (CT2 / out).exists():
        subprocess.check_call([sys.executable, "-m", "ctranslate2.converters.transformers", "--model", hf_id,
                               "--output_dir", str(CT2 / out), "--quantization", "int8", "--force"])
    return str(CT2 / out)

def make_ct2(hf_id, out, src_lang=None, tgt_prefix=None, threads=4):
    import ctranslate2, transformers
    path = convert(hf_id, out)
    tok = transformers.AutoTokenizer.from_pretrained(hf_id, **({"src_lang": src_lang} if src_lang else {}))
    tr = ctranslate2.Translator(path, device="cpu", compute_type="int8", inter_threads=1, intra_threads=threads)
    def run(text):
        toks = tok.convert_ids_to_tokens(tok.encode(text))
        res = tr.translate_batch([toks], target_prefix=[[tgt_prefix]] if tgt_prefix else None, beam_size=2)
        out_toks = res[0].hypotheses[0]
        if tgt_prefix: out_toks = out_toks[1:]
        return tok.decode(tok.convert_tokens_to_ids(out_toks), skip_special_tokens=True)
    return run

MODELS = {
    "opus-mt":      lambda t: make_ct2("Helsinki-NLP/opus-mt-hy-en", "opus-mt-hy-en", threads=t),
    "nllb-600m":    lambda t: make_ct2("facebook/nllb-200-distilled-600M", "nllb-600m",
                                       src_lang="hye_Armn", tgt_prefix="eng_Latn", threads=t),
    "nllb-1.3b":    lambda t: make_ct2("facebook/nllb-200-distilled-1.3B", "nllb-1.3b",
                                       src_lang="hye_Armn", tgt_prefix="eng_Latn", threads=t),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["opus-mt", "nllb-600m"])
    ap.add_argument("--source", choices=["gold", "asr"], default="gold")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    data = load_fleurs(a.n)
    if a.source == "asr":
        from bench_asr import make_fastconformer
        asr = make_fastconformer(True)
        srcs = [asr(d["wav"]) for d in data]
    else:
        srcs = [d["hy"] for d in data]
    refs = [d["en"] for d in data]
    for name in a.models:
        run = MODELS[name](a.threads); run(srcs[0])
        hyps, lat = [], []
        for s in srcs:
            t = time.perf_counter(); hyps.append(run(s)); lat.append(time.perf_counter() - t)
        bleu = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True); chrf = sacrebleu.corpus_chrf([h.lower() for h in hyps], [[r.lower() for r in refs]])
        print(f"== {name} ({a.source} input)\n   BLEU {bleu.score:.1f} | chrF {chrf.score:.1f} | "
              f"latency p50 {pct(lat,.5):.2f}s p95 {pct(lat,.95):.2f}s")
        for i in (1, 2):
            print(f"   SRC {srcs[i][:90]}\n   HYP {hyps[i]}\n   REF {refs[i]}")
        print()

if __name__ == "__main__":
    main()
