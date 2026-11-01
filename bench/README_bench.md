# Armenian ASR + MT benchmarks (CPU)

## Setup (Windows, Python 3.12)
```
pip install onnx-asr onnxruntime ctranslate2 transformers sentencepiece jiwer sacrebleu soundfile huggingface_hub faster-whisper psutil
pip install torch --index-url https://download.pytorch.org/whl/cpu     # only for the one-time NLLB/Marian -> CT2 conversion
python fetch_data.py                       # ~210 MB FLEURS Armenian dev set + English references
python bench_asr.py --models fastconformer-int8 whisper-small --n 40
python bench_mt.py  --models opus-mt nllb-600m --source gold --n 40
python bench_mt.py  --models opus-mt nllb-600m --source asr  --n 40   # ASR output -> MT, the real pipeline
```
Optional: `nllb-1.3b`, `whisper-medium`, `whisper-large-v3-turbo` (expect these to be too slow on 4 CPU cores).

## Caveats
- FLEURS is *read speech* in a quiet room, and the NVIDIA model was trained on FLEURS, so WER is optimistic.
  Real call audio (compressed, overlapping, spontaneous) will be worse. **Record 2-3 minutes of your own Armenian
  speech and a call recording and re-check** - that is the number that matters for the interview.
- Translation scores compare against FLEURS English, which is lowercased/unpunctuated; scoring is case-insensitive.
- Numbers below were measured in a 2-core Linux sandbox, not your PC.
