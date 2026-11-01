"""Build a fake 'call' from Common Voice Armenian clips: one wav for the mic, one for system audio.
Ground truth (who spoke when) is printed so you can check the speaker labels.

  python tools/make_sim_call.py --parquet speaker/data/cv_test.parquet --out sim
  python tools/make_sim_call.py --out hard --gap 0 0.05 --tight --order A,B,A,A,B,B   # back-to-back speakers
"""
import argparse, subprocess, collections
import numpy as np, pyarrow.parquet as pq, soundfile as sf

SR = 16000
def decode(b):
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
                       input=b, capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="speaker/data/cv_test.parquet"); ap.add_argument("--out", default="sim")
ap.add_argument("--turns", type=int, default=12); ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--gap", type=float, nargs=2, default=[0.9, 1.6], help="pause range between turns (s)")
ap.add_argument("--tight", action="store_true", help="trim silence at the start/end of every clip")
ap.add_argument("--order", default="A,Me,B,A,Me,B", help="comma list of speakers (A, B, Me), repeated")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
rows = pq.read_table(a.parquet, columns=["client_id", "audio", "sentence"]).to_pylist()
by = collections.defaultdict(list)
for r in rows: by[r["client_id"]].append(r)
ids = [c for c, v in by.items() if len(v) >= 8]; rng.shuffle(ids)
who = {"Me": ids[0], "Speaker A": ids[1], "Speaker B": ids[2]}
order = [{"A": "Speaker A", "B": "Speaker B"}.get(k, k) for k in a.order.split(",")] * 12
mic, sysc, t = [], [], 0.0
print("Ground truth:")
for name in order[: a.turns]:
    r = by[who[name]].pop()
    x = decode(r["audio"]["bytes"])
    if a.tight:
        nz = np.where(np.abs(x) > 0.02)[0]; x = x[max(0, nz[0] - 800): nz[-1] + 800]
    gap = np.zeros(int(SR * rng.uniform(*a.gap)), np.float32)
    (mic if name == "Me" else sysc).append(x); (sysc if name == "Me" else mic).append(np.zeros_like(x))
    mic.append(gap); sysc.append(gap)
    print(f"  [{int(t)//60:02d}:{int(t)%60:02d}] {name:10s} {r['sentence']}"); t += (len(x) + len(gap)) / SR
sf.write(f"{a.out}_mic.wav", np.concatenate(mic), SR); sf.write(f"{a.out}_system.wav", np.concatenate(sysc), SR)
print(f"\nwrote {a.out}_mic.wav / {a.out}_system.wav ({t:.0f}s)")
