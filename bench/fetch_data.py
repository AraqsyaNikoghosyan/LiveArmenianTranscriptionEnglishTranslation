"""Download a small FLEURS Armenian dev set (+ English references) into bench/data."""
import tarfile, urllib.request
from pathlib import Path
D = Path(__file__).parent / "data"; D.mkdir(exist_ok=True)
B = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/"
for u, f in [(B + "hy_am/dev.tsv", "hy_dev.tsv"), (B + "en_us/dev.tsv", "en_dev.tsv"),
             (B + "hy_am/audio/dev.tar.gz", "fleurs_dev.tar.gz")]:
    if not (D / f).exists():
        print("downloading", f); urllib.request.urlretrieve(u, D / f)
if not any(D.rglob("*.wav")):
    print("extracting"); tarfile.open(D / "fleurs_dev.tar.gz").extractall(D)
print("ok")
