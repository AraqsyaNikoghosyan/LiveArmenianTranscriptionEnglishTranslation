"""Download speaker-embedding models (sherpa-onnx release) + Common Voice Armenian test split."""
import urllib.request
from pathlib import Path
here = Path(__file__).parent
(here / "models").mkdir(exist_ok=True); (here / "data").mkdir(exist_ok=True)
GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
for f in ["nemo_en_titanet_small.onnx",
          "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
          "wespeaker_zh_cnceleb_resnet34.onnx",
          "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"]:
    dst = here / "models" / f
    if not dst.exists() or dst.stat().st_size < 10_000_000:      # re-download missing/truncated files
        print("model", f); urllib.request.urlretrieve(GH + f, dst)
d = here / "data" / "cv_test.parquet"
if not d.exists():
    print("dataset (~200 MB)")
    urllib.request.urlretrieve("https://huggingface.co/datasets/Chillarmo/common_voice_20_armenian/resolve/main/data/test-00000-of-00001.parquet", d)
