"""
Live transcription: listen to a microphone, or to what the computer is playing on any of
its sound cards ("loopback": a video call, a live stream, a video in the browser), and
transcribe it as it is spoken.

The sound is saved as a WAV recording (in the "recordings" folder) as it comes in. When you
stop, the transcribed lines are saved in that recording's folder in "transcripts", like any
other recording's. On the review page the live recording is the open recording: its lines
appear as they are said and can be reviewed and corrected while it is still listening.

How it works: the sound goes, in tenths of a second, through the speech detector (Silero
VAD). Each stretch of speech is transcribed by the main model as soon as it ends (a pause
of about a third of a second), which makes a line. While someone is still speaking, the
stretch so far is transcribed again every half second or so, to show the words as they
come ("partial" text, replaced by the line when the stretch ends).

Listening can be paused (the device is let go; when resumed the recording carries on from
there, without the gap) and moved to another device at any time, in the same recording. On
the review page a device that stops working (unplugged) pauses it, to choose another.
"""

import queue
import struct
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 1600                 # samples captured at a time (0.1 s)
VAD_WINDOW = 512             # samples the speech detector takes at a time
PAD = int(0.15 * SAMPLE_RATE)  # context around each stretch of speech (as for files)
KEEP_SEC = 40                # seconds of sound the speech detector works on (stretches are at most ~15 s)
GROW_SEC = 600               # the whole recording is also kept (for the review page), grown 10 min at a time
PARTIAL_EVERY = 0.5          # seconds between partial transcriptions, at least
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
FLUSH = "flush"              # (in the queue) the sound stops here: paused, or another device


# --------------------------------------------------------------------------- devices
def _soundcard():
    """The soundcard module, ready to use in the calling thread."""
    import sys
    try:
        import soundcard
    except ImportError as e:
        raise RuntimeError("live transcription needs the 'soundcard' package: run setup.bat "
                           "once more to install it") from e
    if sys.platform == "win32":
        # Windows' sound devices are reached through COM, which each thread must start for
        # itself: soundcard does it only in the thread that first imports it, and the review
        # page asks from other threads. (After the import: soundcard takes a COM already
        # started in its thread for an error. Started already here: no harm done.)
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)     # COINIT_MULTITHREADED
    warnings.filterwarnings("ignore", category=getattr(soundcard, "SoundcardRuntimeWarning", RuntimeWarning))
    return soundcard


def list_devices():
    """
    Everything that can be listened to, microphones first:
      [{"id", "name", "kind": "mic" | "loopback", "default": bool, "label"}]
    A "loopback" device is a sound card's output: what the computer plays through it.
    """
    sc = _soundcard()
    try:
        default_mic = sc.default_microphone().id
    except Exception:
        default_mic = None
    try:
        default_spk = sc.default_speaker().id
    except Exception:
        default_spk = None
    out = []
    for m in sc.all_microphones(include_loopback=True):
        loop = bool(getattr(m, "isloopback", False))
        out.append({
            "id": m.id, "name": m.name, "kind": "loopback" if loop else "mic",
            "default": m.id == (default_spk if loop else default_mic),
            "label": (f"What plays on {m.name}" if loop else m.name),
        })
    out.sort(key=lambda d: (d["kind"] != "mic", not d["default"], d["name"].lower()))
    return out


def find_device(choice: str, devices=None):
    """A device from a number in list_devices() (1-based) or part of its name; None if none."""
    devices = devices if devices is not None else list_devices()
    choice = (choice or "").strip()
    if choice.isdigit():
        k = int(choice) - 1
        return devices[k] if 0 <= k < len(devices) else None
    low = choice.lower()
    hits = [d for d in devices if low and (low in d["label"].lower() or d["id"] == choice)]
    return hits[0] if hits else None


def new_recording_path(folder: Path = RECORDINGS_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    path = folder / f"live {stamp}.wav"
    k = 2
    while path.exists():
        path = folder / f"live {stamp} ({k}).wav"
        k += 1
    return path


# --------------------------------------------------------------------------- recording
def wav_header(data_bytes: int) -> bytes:
    """44-byte header of a 16 kHz mono 16-bit WAV."""
    return (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
            + b"data" + struct.pack("<I", data_bytes))


class WavWriter:
    """Writes the sound as it comes; the header is brought up to date every few seconds, so
    even a recording cut short by a crash or a power cut opens with all it has."""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "wb")
        self.f.write(wav_header(0))
        self.bytes = 0
        self.patched = time.time()

    def write(self, x: np.ndarray):
        pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        self.f.write(pcm)
        self.bytes += len(pcm)
        if time.time() - self.patched > 5:
            self._patch()

    def _patch(self):
        pos = self.f.tell()
        self.f.seek(0)
        self.f.write(wav_header(self.bytes))
        self.f.seek(pos)
        self.f.flush()
        self.patched = time.time()

    def close(self):
        if not self.f.closed:
            self._patch()
            self.f.close()


# --------------------------------------------------------------------------- live
class LiveTranscriber:
    """
    One live session: start(), then lines arrive (snapshot() for the page, on_line for the
    console), stop() ends it and returns the lines [(start, dur, [(word, time, conf)])],
    times counted from the start of the recording.

    decode(samples, start_sec) -> [(word, time, conf)] runs the main model on a stretch;
    fix(words) -> words applies the keyword lists (for showing; the saved lines are as
    heard, and the review page applies the keywords itself).
    """

    def __init__(self, device: dict, sherpa_onnx, vad_config, decode, wav_path: Path,
                 fix=None, on_line=None, on_partial=None, pause_on_loss=False):
        self.device = device
        self.sherpa_onnx, self.vad_config = sherpa_onnx, vad_config
        self.decode, self.fix = decode, fix
        self.on_line, self.on_partial = on_line, on_partial
        self.wav_path = wav_path
        self.lock = threading.Lock()
        self.lines = []          # as heard: (start, dur, words)
        self.shown = []          # for the page: {"t", "dur", "text"} (with keyword fixes)
        self.partial = ""        # what is being said now (not final)
        self.partial_t = None    # where it starts
        self.level = 0.0         # loudness of the last tenth of a second, 0..1
        self.loud_at = None      # when there was last some sound
        self.samples = 0         # captured so far
        self._all = np.zeros(0, dtype=np.float32)   # the whole recording (capacity >= samples)
        self.error = None        # why it stopped by itself (device unplugged, ...)
        self.running = False
        self.paused = False      # the device let go until resume()
        self.settled = False     # at a break (paused, another device), with all said so far in lines
        self.notice = None       # why it paused by itself (the device stopped working, ...)
        self.pause_on_loss = pause_on_loss   # (else a device that stops working ends it)
        self.started = None
        self._wake = threading.Event()
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._wav = None
        self._threads = []

    # -- control -------------------------------------------------------------------
    def start(self):
        """Open the device and start; raises if it can't be opened."""
        self._wav = WavWriter(self.wav_path)
        self.started = time.time()
        self.running = True
        cap = threading.Thread(target=self._capture, daemon=True, name="live-capture")
        work = threading.Thread(target=self._process, daemon=True, name="live-transcribe")
        self._threads = [cap, work]
        cap.start()
        work.start()
        self._ready.wait(10)
        if self.error:
            self.stop()
            self.wav_path.unlink(missing_ok=True)
            raise RuntimeError(self.error)
        return self

    def stop(self):
        """Stop listening, transcribe what is left, close the recording; -> lines."""
        self._stop.set()
        for t in self._threads:
            t.join(60)
        if self._wav:
            self._wav.close()
        with self.lock:
            self.running = False
            self.partial, self.partial_t = "", None
        return list(self.lines)

    def pause(self):
        """Let go of the device; what is being said is finished as a line. resume() goes on."""
        with self.lock:
            self.paused = True
        self._wake.set()

    def resume(self, device: dict = None):
        """Listen again (to `device` from now on, if given), in the same recording."""
        with self.lock:
            if device:
                self.device = device
            self.paused, self.notice = False, None
            self.loud_at = time.time()
        self._wake.set()

    def set_device(self, device: dict):
        """Listen to another device from now on (if paused: once resumed)."""
        with self.lock:
            self.device, self.notice = device, None
            self.loud_at = time.time()
        self._wake.set()

    @property
    def duration(self) -> float:
        return self.samples / SAMPLE_RATE

    def audio(self) -> np.ndarray:
        """The recording so far (a view: no copy; it stays valid as the recording grows)."""
        with self.lock:
            return self._all[:self.samples]

    def behind(self) -> float:
        """Seconds of sound captured but not yet through the speech detector."""
        return self._q.qsize() * BLOCK / SAMPLE_RATE

    def lines_from(self, k: int):
        with self.lock:
            return self.lines[k:]

    def snapshot(self, first: int = 0) -> dict:
        with self.lock:
            now = time.time()
            return {
                "running": self.running, "device": self.device.get("label", ""),
                "device_id": self.device.get("id"), "paused": self.paused, "notice": self.notice,
                "kind": self.device.get("kind", "mic"), "path": str(self.wav_path),
                "file": self.wav_path.name, "elapsed": round(self.duration, 1),
                "level": round(self.level, 3),
                "quiet_for": round(now - (self.loud_at or self.started or now), 1),
                "behind": round(self.behind(), 1),
                "partial": self.partial, "partial_t": self.partial_t,
                "n": len(self.shown), "lines": self.shown[first:], "error": self.error,
            }

    # -- capture (its own thread, so no sound is lost while the model works) ----------
    def _capture(self):
        try:
            sc = _soundcard()
            while not self._stop.is_set():
                if self.paused:
                    self._wake.wait(0.5)
                    self._wake.clear()
                    continue
                dev = self.device
                try:
                    rec = sc.get_microphone(dev["id"], include_loopback=True).recorder(
                        samplerate=SAMPLE_RATE, blocksize=BLOCK)
                    rec.__enter__()
                except Exception as e:
                    if not self._ready.is_set():
                        raise
                    self._lost(dev, e, opening=True)
                    continue
                self._ready.set()
                try:
                    # Until stopped, paused, or moved to another device.
                    while not self._stop.is_set() and not self.paused and self.device is dev:
                        data = rec.record(numframes=BLOCK)
                        x = (data.mean(axis=1) if data.ndim > 1 else data).astype(np.float32)
                        if len(x):
                            self._take(x)
                except Exception as e:
                    self._lost(dev, e)
                finally:
                    try:
                        rec.__exit__(None, None, None)
                    except Exception:
                        pass
                    self._q.put(FLUSH)
        except Exception as e:
            msg = str(e) or e.__class__.__name__
            with self.lock:
                self.error = (f"Can't listen to {self.device.get('label', 'this device')}: {msg}"
                              if not self._ready.is_set() else f"Listening stopped: {msg}")
            print(f"\n  live: {self.error}")
            self._ready.set()
        finally:
            self._q.put(None)      # the end, for the transcribing thread

    def _take(self, x: np.ndarray):
        self._wav.write(x)
        self._q.put(x)
        rms = float(np.sqrt(np.mean(x * x)))
        db = 20 * np.log10(max(rms, 1e-6))
        with self.lock:
            n = self.samples + len(x)
            if n > len(self._all):          # (views given out keep the old array)
                grown = np.zeros(len(self._all) + GROW_SEC * SAMPLE_RATE, dtype=np.float32)
                grown[:self.samples] = self._all[:self.samples]
                self._all = grown
            self._all[self.samples:n] = x
            self.samples = n
            self.level = max(0.0, min(1.0, (db + 60) / 60))   # -60 dB .. 0 dB
            if db > -50:
                self.loud_at = time.time()

    def _lost(self, dev: dict, e: Exception, opening=False):
        """The device can't be opened, or stopped working: pause, to choose another
        (without pause_on_loss, as in the terminal: the session ends)."""
        if not self.pause_on_loss:
            raise e
        msg = str(e) or e.__class__.__name__
        label = dev.get("label", "this device")
        note = f"Can't listen to {label} ({msg})." if opening else f"{label} stopped working ({msg})."
        print(f"\n  live: {note} Paused.")
        with self.lock:
            self.paused, self.level = True, 0.0
            self.notice = note + " Choose a device, then Resume."

    # -- speech detection and transcription --------------------------------------------
    def _process(self):
        vad = self.sherpa_onnx.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=120)
        buf = np.zeros(0, dtype=np.float32)   # the last KEEP_SEC seconds of sound
        buf0 = 0                               # sample number of buf[0]
        pending = np.zeros(0, dtype=np.float32)  # not yet given to the speech detector
        fed = 0                                # samples given to the speech detector
        last_end = 0                           # where the last line ended
        speech_from = None                     # where the current stretch began (about)
        last_partial, cost = 0.0, 0.1
        vad0 = 0                               # samples given before this speech detector (a break)

        def cut(a, b):
            a, b = max(a, buf0), min(b, buf0 + len(buf))
            return buf[a - buf0:b - buf0] if b > a else np.zeros(0, dtype=np.float32)

        def drain():
            nonlocal last_end, speech_from
            while not vad.empty():
                seg = vad.front
                s = vad0 + seg.start
                e = s + len(seg.samples)
                vad.pop()
                a = max(s - PAD, vad0)          # (not into the sound before a break)
                samples = cut(a, e + PAD)
                if len(samples):
                    self._final(max(a, buf0), samples)
                last_end, speech_from = e, None

        def finish():
            """At a break (paused, another device): what was being said is a line now."""
            nonlocal vad, vad0, pending, fed, speech_from
            if len(pending):
                vad.accept_waveform(pending)
                fed += len(pending)
                pending = np.zeros(0, dtype=np.float32)
            vad.flush()
            drain()
            vad = self.sherpa_onnx.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=120)
            vad0, speech_from = fed, None
            with self.lock:
                self.partial, self.partial_t = "", None
            if self.on_partial:
                self.on_partial("")

        try:
            while True:
                try:
                    x = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if x is None:
                    break
                if isinstance(x, str):          # FLUSH
                    finish()
                    self.settled = True
                    continue
                self.settled = False
                buf = np.concatenate([buf, x])
                if len(buf) > KEEP_SEC * SAMPLE_RATE:
                    drop = len(buf) - KEEP_SEC * SAMPLE_RATE
                    buf, buf0 = buf[drop:], buf0 + drop
                pending = np.concatenate([pending, x])
                n = len(pending) // VAD_WINDOW * VAD_WINDOW
                for i in range(0, n, VAD_WINDOW):
                    vad.accept_waveform(pending[i:i + VAD_WINDOW])
                pending, fed = pending[n:], fed + n
                drain()

                # The words so far of what is being said now, if keeping up.
                if vad.is_speech_detected():
                    if speech_from is None:
                        speech_from = max(last_end, vad0, fed - int(0.8 * SAMPLE_RATE))
                    if (self._q.qsize() <= 2 and
                            time.time() - last_partial > max(PARTIAL_EVERY, 3 * cost)):
                        t0 = time.time()
                        now = buf0 + len(buf)
                        a = max(speech_from - PAD, vad0, now - int(self.vad_config.silero_vad.max_speech_duration * SAMPLE_RATE))
                        words = self.decode(cut(a, now), max(a, buf0) / SAMPLE_RATE)
                        cost = time.time() - t0
                        last_partial = time.time()
                        self._set_partial(words, max(a, buf0) / SAMPLE_RATE)
            if len(pending):
                vad.accept_waveform(pending)
            vad.flush()
            drain()
        except Exception as e:   # (shouldn't happen; what's recorded is kept)
            with self.lock:
                self.error = f"Transcribing stopped: {e}"
            print(f"\n  live: {self.error}")
            self._stop.set()

    def _set_partial(self, words, t):
        text = self._text(words)
        with self.lock:
            self.partial, self.partial_t = text, (round(t, 2) if text else None)
        if self.on_partial:
            self.on_partial(text)

    def _text(self, words):
        if self.fix and words:
            words = self.fix(words)
        return " ".join(w[0] for w in words)

    def _final(self, start_sample: int, samples: np.ndarray):
        start, dur = start_sample / SAMPLE_RATE, len(samples) / SAMPLE_RATE
        words = self.decode(samples, start)
        text = self._text(words) if words else ""
        with self.lock:
            if words:
                self.lines.append((start, dur, words))
                self.shown.append({"t": round(start, 2), "dur": round(dur, 2), "text": text})
            self.partial, self.partial_t = "", None
        if words and self.on_line:
            self.on_line(start, text)
