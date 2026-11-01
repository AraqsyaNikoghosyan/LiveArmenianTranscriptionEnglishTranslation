"""Shared helpers for the Armenian ASR / MT benchmarks."""
import csv, re, sys, unicodedata
from pathlib import Path

DATA = Path(__file__).parent / "data"
csv.field_size_limit(10**7)


def load_fleurs(n=40, split="dev"):
    """Return up to n dicts {wav, hy, en} from FLEURS hy_am, with the English
    reference found by matching the sentence id in the en_us tsv."""
    def rows(p):
        with open(p, encoding="utf-8", newline="") as f:
            return list(csv.reader(f, delimiter="\t"))
    en = {}
    for r in rows(DATA / f"en_{split}.tsv"):
        en.setdefault(r[0], r[3])          # col0 = sentence id, col3 = normalised text
    wav_dir = next(DATA.rglob(f"{split}"), None)
    out, seen = [], set()
    for r in rows(DATA / f"hy_{split}.tsv"):
        sid, fn, raw = r[0], r[1], r[2]
        if sid in seen or sid not in en:   # one speaker per sentence
            continue
        wav = next(DATA.rglob(fn), None)
        if wav is None:
            continue
        seen.add(sid)
        out.append({"wav": str(wav), "hy": raw, "en": en[sid]})
        if len(out) >= n:
            break
    if not out:
        sys.exit("No FLEURS data found in bench/data - run: python fetch_data.py")
    return out


def norm_hy(s):
    """Normalise Armenian text for WER/CER: lowercase, unify ew ligature, drop punctuation."""
    s = unicodedata.normalize("NFC", s).lower()
    s = s.replace("և", "եւ")
    s = re.sub(r"[^\w\s]", " ", s)           # \w keeps Armenian letters
    return re.sub(r"\s+", " ", s).strip()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]
