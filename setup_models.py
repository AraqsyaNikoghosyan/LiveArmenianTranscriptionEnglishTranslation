"""One-time model setup: downloads VAD + speaker models, converts the translation model to CTranslate2 int8.
Weights are NOT part of the repository - run this after cloning."""
import subprocess, sys, urllib.request
from pathlib import Path

M = Path(__file__).parent / "models"; M.mkdir(exist_ok=True)
GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
FILES = {
    "silero_vad.onnx": GH + "asr-models/silero_vad.onnx",
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx":
        GH + "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
}
for name, url in FILES.items():
    dst = M / name
    if not dst.exists() or dst.stat().st_size < 500_000:
        print("downloading", name); urllib.request.urlretrieve(url, dst)

seg = M / "sherpa-onnx-pyannote-segmentation-3-0"
if not (seg / "model.int8.onnx").exists():
    import tarfile
    print("downloading speaker-change segmentation model")
    tgz = M / "seg.tar.bz2"
    urllib.request.urlretrieve(GH + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", tgz)
    tarfile.open(tgz).extractall(M); tgz.unlink()

if not (M / "nllb-600m-ct2" / "model.bin").exists():
    print("converting NLLB-200-distilled-600M to CTranslate2 int8 (needs torch; downloads ~2.4 GB once) ...")
    subprocess.check_call([sys.executable, "-m", "ctranslate2.converters.transformers",
                           "--model", "facebook/nllb-200-distilled-600M",
                           "--output_dir", str(M / "nllb-600m-ct2"), "--quantization", "int8", "--force"])

print("Fetching the Armenian speech model ...")
from huggingface_hub import snapshot_download
snapshot_download("Vana-Labs/stt-fastconformer-armenian", allow_patterns=["*.json", "*.txt", "nemo80.onnx", "*.int8.onnx"])
print("Done. Next: python -m live --list-devices")
