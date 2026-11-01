"""python -m live [--list-devices] [--mic ID] [--loopback ID]"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(prog="python -m live", description="Live Armenian -> English call translator (local)")
    ap.add_argument("--list-devices", action="store_true", help="print microphones and loopback devices, then exit")
    ap.add_argument("--mic", type=int, help="microphone device id (default: Windows default input)")
    ap.add_argument("--loopback", type=int, help="system-audio loopback device id (default: loopback of default output)")
    ap.add_argument("--no-mic", action="store_true", help="only listen to system audio")
    ap.add_argument("--no-system", action="store_true", help="only listen to the microphone")
    ap.add_argument("--sim", nargs=2, metavar=("MIC_WAV", "SYSTEM_WAV"),
                    help="test without audio devices: play two wav files as mic / system audio")
    ap.add_argument("--speed", type=float, default=1.0, help="with --sim: playback speed")
    ap.add_argument("--spk-threshold", type=float, default=0.45, help="speaker similarity threshold (lower = fewer speakers)")
    ap.add_argument("--min-silence", type=float, default=0.45, help="pause (s) that ends a phrase")
    ap.add_argument("--max-speech", type=float, default=8.0, help="cut phrases longer than this (s)")
    ap.add_argument("--split-thr", type=float, default=0.30,
                    help="split a phrase when two speakers talk back-to-back (higher = split more eagerly, 0 = off)")
    ap.add_argument("--splitter", choices=["embed", "pyannote"], default="pyannote", help="how to find speaker changes inside a phrase")
    ap.add_argument("--beam", type=int, default=1, help="translation beam size (1 = fastest)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--no-split", action="store_true", help="do not split a phrase when the speaker changes inside it")
    ap.add_argument("--dump", metavar="DIR", help="save every recognised phrase as wav + txt (to debug recognition)")
    ap.add_argument("--sim-loopback", action="store_true", help="with --sim: behave like WASAPI loopback (no frames in silence)")
    ap.add_argument("--show-latency", action="store_true", help="print end-of-speech -> output delay")
    ap.add_argument("--models", default=str(ROOT / "models"), help="folder created by setup_models.py")
    a = ap.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")          # Armenian in the Windows console

    if a.list_devices:
        from .audio import list_devices; list_devices(); return

    from .audio import FileSource, WasapiSource, default_ids
    from .pipeline import Pipeline, Settings
    sources = {}
    if a.sim:
        sources["mic"] = FileSource(a.sim[0], "sim-mic", a.speed)
        sources["system"] = FileSource(a.sim[1], "sim-system", a.speed, loopback_like=a.sim_loopback)
    else:
        dmic, dloop = default_ids()
        if not a.no_mic:
            sources["mic"] = WasapiSource(a.mic if a.mic is not None else dmic, "mic")
        if not a.no_system:
            idx = a.loopback if a.loopback is not None else dloop
            if idx is None: sys.exit("No loopback device found. Run --list-devices.")
            sources["system"] = WasapiSource(idx, "system")
    if not sources: sys.exit("Nothing to listen to.")

    m = Path(a.models)
    cfg = Settings(vad_model=str(m / "silero_vad.onnx"),
                   speaker_model=str(m / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"),
                   translate_dir=str(m / "nllb-600m-ct2"),
                   segmentation_model=str(m / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx"),
                   split_turns=not a.no_split, splitter=a.splitter, dump_dir=a.dump or "", spk_threshold=a.spk_threshold,
                   min_silence=a.min_silence, max_speech=a.max_speech, split_thr=a.split_thr, beam=a.beam,
                   show_latency=a.show_latency, threads=a.threads)
    Pipeline(sources, cfg).run()


main()
