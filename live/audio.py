"""Audio sources: Windows WASAPI (mic + loopback of the speakers) and a file simulator.

Every source yields mono float32 chunks at 16 kHz through `read()` / a callback, so the rest
of the pipeline never has to care where the audio came from.
"""
from __future__ import annotations
import math, queue, threading, time
import numpy as np
from scipy.signal import resample_poly

SR = 16000


# --------------------------------------------------------------------------- devices
def _pa():
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as e:                      # pragma: no cover
        raise SystemExit("PyAudioWPatch is missing (Windows only): pip install PyAudioWPatch") from e
    return pyaudio, pyaudio.PyAudio()


def list_devices():
    """Print WASAPI microphones and loopback (system-audio) devices with their ids."""
    pyaudio, p = _pa()
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    d_in, d_out = wasapi["defaultInputDevice"], wasapi["defaultOutputDevice"]
    print("Microphones (use with --mic):")
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d["hostApi"] == wasapi["index"] and d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice"):
            print(f"  [{i:2d}] {d['name']}{'   <- default' if i == d_in else ''}")
    print("\nSystem audio / loopback (use with --loopback):")
    for d in p.get_loopback_device_info_generator():
        mark = "   <- default output" if d["name"].startswith(p.get_device_info_by_index(d_out)["name"]) else ""
        print(f"  [{d['index']:2d}] {d['name']}{mark}")
    p.terminate()


def default_ids():
    pyaudio, p = _pa()
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    mic = wasapi["defaultInputDevice"]
    out_name = p.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
    loop = next((d["index"] for d in p.get_loopback_device_info_generator() if d["name"].startswith(out_name)), None)
    p.terminate()
    return mic, loop


# --------------------------------------------------------------------------- sources
class Source:
    """Base: produces 16 kHz mono float32 chunks into a queue (None = end of stream)."""
    name = "?"
    pads_silence = False     # True for WASAPI: it delivers NO frames while nothing plays, so the VAD must be fed silence
    def __init__(self):
        self.q: "queue.Queue[tuple[np.ndarray, float] | None]" = queue.Queue()   # (chunk, wall-clock arrival)
        self.t0 = None                       # wall-clock time of the first sample
    def start(self): raise NotImplementedError
    def stop(self): pass


class WasapiSource(Source):
    pads_silence = True
    def __init__(self, device_index: int, name: str):
        super().__init__()
        self.index, self.name = device_index, name
        self._raw: "queue.Queue[bytes]" = queue.Queue()
        self._run = True

    def start(self):
        pyaudio, self.p = _pa()
        d = self.p.get_device_info_by_index(self.index)
        self.rate, self.ch = int(d["defaultSampleRate"]), int(d["maxInputChannels"])
        g = math.gcd(self.rate, SR); self.up, self.down = SR // g, self.rate // g

        def cb(data, frames, tinfo, status):              # runs on PortAudio's thread: keep it tiny
            self._raw.put((data, time.time())); return (None, pyaudio.paContinue)

        self.stream = self.p.open(format=pyaudio.paInt16, channels=self.ch, rate=self.rate, input=True,
                                  input_device_index=self.index, frames_per_buffer=int(self.rate * 0.05),
                                  stream_callback=cb)
        self.t0 = time.time()
        threading.Thread(target=self._convert, daemon=True).start()

    def _convert(self):
        # NOTE: WASAPI loopback sends NO frames while nothing is playing; Pipeline._vad_loop fills those gaps
        # with silence (pads_silence = True), so sample time keeps following the wall clock.
        while self._run:
            try:
                data, wall = self._raw.get(timeout=0.5)
            except queue.Empty:
                continue
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            x = x.reshape(-1, self.ch).mean(axis=1)               # down-mix to mono
            if self.rate != SR:
                x = resample_poly(x, self.up, self.down).astype(np.float32)
            self.q.put((x, wall))

    def stop(self):
        self._run = False
        try: self.stream.stop_stream(); self.stream.close(); self.p.terminate()
        except Exception: pass
        self.q.put(None)


class FileSource(Source):
    """Plays a wav file into the pipeline at real-time speed (or faster) - for tests without audio devices."""
    def __init__(self, path: str, name: str, speed: float = 1.0, start_delay: float = 0.0, loopback_like: bool = False):
        super().__init__()
        self.path, self.name, self.speed, self.delay = path, name, speed, start_delay
        self.loopback_like = self.pads_silence = loopback_like   # drop all-zero chunks, like WASAPI loopback does

    def start(self):
        import soundfile as sf
        x, sr = sf.read(self.path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR: x = resample_poly(x, SR, sr).astype(np.float32)
        self.t0 = time.time() + self.delay
        threading.Thread(target=self._play, args=(x,), daemon=True).start()

    def _play(self, x):
        time.sleep(self.delay)
        step = int(SR * 0.1)
        t = time.time()
        for i in range(0, len(x), step):
            if not (self.loopback_like and not x[i:i + step].any()):
                self.q.put((x[i:i + step], time.time()))
            t += 0.1 / self.speed
            time.sleep(max(0, t - time.time()))
        self.q.put(None)
