"""Benchmark speaker-embedding models for online speaker labelling on Armenian speech.

Data: Common Voice 20 Armenian (client_id = speaker id). Builds simulated "calls" with K speakers,
runs OnlineSpeakerTracker, reports accuracy (best label mapping), #speakers found, embed latency.

  python bench_speakers.py --models models/*.onnx --degrade opus
"""
import argparse, glob, io, subprocess, time, itertools, collections
import numpy as np, pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from speakers import Embedder, OnlineSpeakerTracker

SR = 16000

def decode(b, degrade=None):
    """bytes -> float32 16k mono; optionally pass through a lossy codec (call/YouTube-like)."""
    cmd = ["ffmpeg", "-v", "quiet", "-i", "pipe:0"]
    if degrade == "opus":        # Telegram-style voice: Opus ~16 kbps
        p = subprocess.run(cmd + ["-ar", "16000", "-ac", "1", "-c:a", "libopus", "-b:a", "16k", "-f", "ogg", "pipe:1"], input=b, capture_output=True)
        b = p.stdout
        cmd = ["ffmpeg", "-v", "quiet", "-i", "pipe:0"]
    p = subprocess.run(cmd + ["-ar", str(SR), "-ac", "1", "-f", "f32le", "pipe:1"], input=b, capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()

def trim_silence(x, thr=0.01, pad=800):
    idx = np.where(np.abs(x) > thr)[0]
    return x[max(0, idx[0] - pad): idx[-1] + pad] if len(idx) else x

def load_clips(n_spk, per_spk, degrade, seed):
    rng = np.random.default_rng(seed)
    t = pq.read_table("data/cv_test.parquet", columns=["client_id", "audio", "gender"]).to_pylist()
    by = collections.defaultdict(list)
    for r in t: by[r["client_id"]].append(r)
    ok = [c for c, v in by.items() if len(v) >= per_spk]
    rng.shuffle(ok); ok = ok[:n_spk]
    clips = []
    for c in ok:
        for r in rng.permutation(len(by[c]))[:per_spk]:
            x = trim_silence(decode(by[c][r]["audio"]["bytes"], degrade))
            clips.append((c, by[c][r]["gender"], x))
    return clips

def accuracy(true, pred):
    ts, ps = sorted(set(true)), sorted(set(pred))
    m = np.zeros((len(ts), len(ps)))
    for t_, p_ in zip(true, pred): m[ts.index(t_), ps.index(p_)] += 1
    r, c = linear_sum_assignment(-m)
    return m[r, c].sum() / len(true)

def simulate(embs, durs, labels, rng, k, thr, min_dur, n_trials=30):
    """Random call: pick k speakers, interleave their clips in random turn order."""
    spk = sorted(set(labels)); accs, found = [], []
    for _ in range(n_trials):
        chosen = list(rng.choice(spk, k, replace=False))
        idx = [i for i, l in enumerate(labels) if l in chosen]
        rng.shuffle(idx)
        tr = OnlineSpeakerTracker(threshold=thr, min_dur=min_dur)
        pred = [tr.assign(embs[i], durs[i]) for i in idx]
        accs.append(accuracy([labels[i] for i in idx], pred)); found.append(len(tr.sums))
    return np.mean(accs), np.mean(found)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=sorted(glob.glob("models/*.onnx")))
    ap.add_argument("--pool", type=int, default=14, help="speakers in pool")
    ap.add_argument("--per", type=int, default=12, help="clips per speaker")
    ap.add_argument("--degrade", choices=["none", "opus"], default="none")
    ap.add_argument("--max-dur", type=float, default=0, help="crop clips to N seconds (0 = no crop)")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    clips = load_clips(a.pool, a.per, None if a.degrade == "none" else a.degrade, a.seed)
    if a.max_dur: clips = [(c, g, x[: int(a.max_dur * SR)]) for c, g, x in clips]
    labels = [c for c, _, _ in clips]; durs = [len(x) / SR for _, _, x in clips]
    print(f"{len(clips)} clips, {a.pool} speakers, mean {np.mean(durs):.1f}s, degrade={a.degrade}\n")
    for mp in a.models:
        emb = Embedder(mp, a.threads); emb(clips[0][2])
        lat, E = [], []
        for _, _, x in clips:
            t = time.perf_counter(); E.append(emb(x)); lat.append(time.perf_counter() - t)
        E = np.array(E)
        same = [E[i] @ E[j] for i, j in itertools.combinations(range(len(E)), 2) if labels[i] == labels[j]]
        diff = [E[i] @ E[j] for i, j in itertools.combinations(range(len(E)), 2) if labels[i] != labels[j]]
        print(f"== {mp.split('/')[-1]}  embed p50 {np.median(lat)*1000:.0f} ms | cos same-spk {np.mean(same):.2f} "
              f"diff-spk {np.mean(diff):.2f}")
        best = None
        for thr in np.arange(0.2, 0.75, 0.05):
            row = []
            for k in (2, 3, 4):
                acc, nf = simulate(E, durs, labels, np.random.default_rng(1), k, thr, a.min_dur)
                row.append((acc, nf))
            score = np.mean([r[0] for r in row])
            if best is None or score > best[0]: best = (score, thr, row)
            print(f"   thr {thr:.2f}: " + " | ".join(f"k={k} acc {r[0]*100:4.0f}% found {r[1]:.1f}" for k, r in zip((2, 3, 4), row)))
        print(f"   -> best thr {best[1]:.2f}, mean acc {best[0]*100:.0f}%\n")

if __name__ == "__main__":
    main()
