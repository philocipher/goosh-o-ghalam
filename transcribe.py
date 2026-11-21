"""
Goosh-o-Ghalam (گوش و قلم, "ear and pen"): Persian (Farsi) audio/video -> .srt subtitles
and transcripts, fully offline.

Model : Shenava Koochik v1.0 (114M, FastConformer CTC), run with onnxruntime
VAD   : Silero VAD via sherpa-onnx (splits speech into subtitle-sized segments)

Usage:
    python transcribe.py interview.mp3
    python transcribe.py video.mp4 another.m4a --out-dir subs
    python transcribe.py "D:\\recordings"            (every audio/video file in the folder)

Names / special terms: list them in keywords.txt (see that file) to fix their spelling.

The first run downloads the model (~460 MB) into the "models" folder next to this script.
"""

import argparse
import shutil
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

import workdir

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 16000

MODEL_NAME = "Shenava-Koochik-v1.0"
MODEL_TITLE = "Shenava Koochik v1.0"   # the speech-recognition model that makes the transcript
MODEL_ABOUT = "FastConformer speech recognition for Persian · 114M parameters"
MODEL_URL = "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0"
HF = f"https://huggingface.co/Reza2kn/{MODEL_NAME}-sherpa-onnx/resolve/main"
MODEL_FILES = {
    "model.onnx": f"{HF}/model.onnx",
    "tokens.txt": f"{HF}/tokens.txt",
    "persian_itn.py": f"{HF}/persian_itn.py",
    "silero_vad.onnx": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
}

MEDIA_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr", ".webm",
    ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".3gp", ".mpeg", ".mpg",
}

# Subtitle layout
MAX_LINE_CHARS = 42      # characters per subtitle line
MAX_LINES = 2            # lines per subtitle cue
MAX_CUE_SECONDS = 7.0    # longest time one cue stays on screen

# Written next to the .srt when off-topic stretches are marked on the review page
LESSON_FILES = (".lesson.srt", ".lesson.txt", ".offtopic.csv")


# --------------------------------------------------------------------------- setup
def models_missing(model_dir: Path):
    return [n for n in MODEL_FILES
            if not ((model_dir / n).exists() and (model_dir / n).stat().st_size > 0)]


def download_models(model_dir: Path, status=None) -> None:
    """
    The main model and the speech detector, if not there yet (about 460 MB, first time only).
    status(text, fraction) reports progress (on the review page); otherwise it's printed.
    """
    missing = models_missing(model_dir)
    for k, name in enumerate(missing, 1):
        what = ("the speech detector" if name == "silero_vad.onnx"
                else f"the speech-recognition model {MODEL_TITLE}")
        last = [0.0]

        def progress(done, total, name=name, what=what, k=k):
            if time.time() - last[0] < 0.3 and done != total:
                return
            last[0] = time.time()
            frac = done / total if total else None
            size = f"{total / 1e6:.0f} MB" if total else f"{done / 1e6:.0f} MB so far"
            text = f"Setting up ({k} of {len(missing)}): downloading {what}, {name} ({size})"
            if frac is not None:
                text += f"… {frac:.0%}"
            if status:
                status(text, frac)
            else:
                print(f"\r  {text}", end="", flush=True)

        if not status:
            print(f"Downloading {name} ...")
        workdir.fetch(MODEL_FILES[name], model_dir / name, progress)
        if not status:
            print()


def load_audio(path: Path, progress=None) -> np.ndarray:
    """
    Decode any audio/video file to 16 kHz mono float32 using PyAV (bundled FFmpeg).
    progress(fraction 0..1) is called now and then while decoding, if given.
    """
    import av

    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("file has no audio track")
        stream = container.streams.audio[0]
        total = (float(container.duration) / av.time_base) if container.duration else 0.0
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        last = 0.0
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
            if progress and total and frame.time is not None and time.time() - last > 0.3:
                last = time.time()
                progress(min(1.0, max(0.0, frame.time / total)))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


# --------------------------------------------------------------------------- core
class Recognizer:
    """
    Runs the CTC model with onnxruntime and decodes it greedily.

    This replaces sherpa-onnx's recognizer, which dropped the last word or ending of many
    segments (e.g. «توضیح بد» for «توضیح بدین», and most final «؟»). It is as fast, and
    it also gives each token's probability, used by --review / --blank-uncertain.
    """

    def __init__(self, model_dir: Path, threads: int):
        import kaldi_native_fbank as knf
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.frame_sec = 0.01 * int(meta.get("subsampling_factor", 8))

        self.tokens = {}
        for line in (model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines():
            if " " in line:
                tok, idx = line.rsplit(" ", 1)
                self.tokens[int(idx)] = tok.replace("▁", " ")
        self.blank = max(self.tokens) if self.tokens.get(max(self.tokens)) == "<blk>" else None

        # Features as NeMo FastConformer models expect them (same as sherpa-onnx).
        self.knf = knf
        o = knf.FbankOptions()
        o.frame_opts.dither = 0
        o.frame_opts.samp_freq = SAMPLE_RATE
        o.frame_opts.snip_edges = False
        o.frame_opts.remove_dc_offset = False
        o.frame_opts.window_type = "hann"
        o.mel_opts.num_bins = int(meta.get("feat_dim", 80))
        o.mel_opts.low_freq = 0
        o.mel_opts.high_freq = 0
        o.mel_opts.is_librosa = True
        o.mel_opts.norm = "slaney"
        self.fbank_opts = o

    def log_probs(self, samples: np.ndarray) -> np.ndarray:
        fb = self.knf.OnlineFbank(self.fbank_opts)
        fb.accept_waveform(SAMPLE_RATE, samples.tolist())
        fb.input_finished()
        n = fb.num_frames_ready
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32)
        feats = np.stack([np.asarray(fb.get_frame(i), dtype=np.float32) for i in range(n)])
        lp, ln = self.sess.run(None, {"audio_signal": feats.T[None],
                                      "length": np.array([n], dtype=np.int64)})
        return lp[0, :int(ln[0])]

    def decode(self, samples: np.ndarray):
        """-> object with .text, .tokens, .timestamps (seconds) and .confidences (0..1)."""
        from types import SimpleNamespace

        lp = self.log_probs(samples)
        blank = self.blank if self.blank is not None else lp.shape[1] - 1
        ids = lp.argmax(axis=1) if len(lp) else np.zeros(0, dtype=np.int64)
        tokens, stamps, conf = [], [], []
        prev = -1
        for f, i in enumerate(ids):
            i = int(i)
            if i != blank and i != prev:
                tokens.append(self.tokens.get(i, ""))
                stamps.append(f * self.frame_sec)
                # the token's best probability within a frame of where it was emitted
                conf.append(float(np.exp(lp[max(0, f - 1):f + 2, i].max())))
            prev = i
        return SimpleNamespace(text="".join(tokens).strip(), tokens=tokens,
                               timestamps=stamps, confidences=conf)


def build_models(model_dir: Path, threads: int, max_segment: float):
    import sherpa_onnx

    recognizer = Recognizer(model_dir, threads)
    vad_config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(model_dir / "silero_vad.onnx"),
            threshold=0.5,
            min_silence_duration=0.35,
            min_speech_duration=0.25,
            max_speech_duration=max_segment,
            window_size=512,
        ),
        sample_rate=SAMPLE_RATE,
        num_threads=1,
    )
    return sherpa_onnx, recognizer, vad_config


def vad_segments(sherpa_onnx, vad_config, audio: np.ndarray, from_sec: float = 0.0):
    """
    Yield (start_seconds, samples) for every speech region, as soon as each is found, so
    transcribing can start before the whole file has been scanned. from_sec skips what
    an earlier, interrupted run already did.
    """
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=120)
    window = vad_config.silero_vad.window_size
    pad = int(0.15 * SAMPLE_RATE)  # a little context around each segment helps accuracy
    # on the same 512-sample grid as a run from the start, and never before from_sec
    base = min(len(audio), -(-int(from_sec * SAMPLE_RATE) // window) * window)

    def drain():
        found = []
        while not vad.empty():
            seg = vad.front
            s = max(base, base + seg.start - pad)
            e = min(len(audio), base + seg.start + len(seg.samples) + pad)
            found.append((s / SAMPLE_RATE, audio[s:e]))
            vad.pop()
        return found

    for i in range(base, len(audio), window):
        vad.accept_waveform(audio[i:i + window])
        yield from drain()
    vad.flush()
    yield from drain()


def words_with_times(result, seg_start: float, seg_dur: float, token_conf=None):
    """
    Turn token-level CTC output into [(word, start_sec)] (a token starting with a space/'▁'
    begins a word). With token_conf, returns [(word, start_sec, confidence)] where a word's
    confidence is that of its least certain token.
    """
    tokens = list(getattr(result, "tokens", []) or [])
    stamps = list(getattr(result, "timestamps", []) or [])
    if token_conf is not None and len(token_conf) != len(tokens):
        token_conf = None
    if tokens and len(tokens) == len(stamps):
        words = []
        for k, (tok, ts) in enumerate(zip(tokens, stamps)):
            t = seg_start + float(ts)
            c = token_conf[k] if token_conf is not None else 1.0
            if tok[:1] in ("▁", " ") or not words:  # word boundary
                piece = tok.lstrip("▁ ")
                if piece:
                    words.append([piece, t, c])
                elif not words or words[-1][0]:
                    words.append(["", t, c])
            else:
                words[-1][0] += tok
                words[-1][2] = min(words[-1][2], c)
        words = [(w, t, c) for w, t, c in words if w]
        if words:
            return words if token_conf is not None else [(w, t) for w, t, _ in words]
    # Fallback: spread words evenly across the segment.
    text_words = result.text.split()
    n = max(1, len(text_words))
    words = [(w, seg_start + seg_dur * i / n) for i, w in enumerate(text_words)]
    return words if token_conf is None else [(w, t, 1.0) for w, t in words]


# --------------------------------------------------------------------------- keywords
# The CTC model can't be biased toward words while decoding (sherpa-onnx only supports
# hotwords for transducer models), so keywords are applied as a correction pass on the
# recognised words: near-misses of a keyword are replaced by its correct spelling.

_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "أ": "ا", "إ": "ا", "ٱ": "ا", "ۀ": "ه",
    "ـ": None,  # tatweel
    **{chr(c): None for c in range(0x064B, 0x0653)},  # short-vowel marks
})
# Letters that sound the same in Persian; swapping them is a cheap edit.
_SOUND_GROUPS = ["زذضظ", "سصث", "تط", "هح", "قغ", "اآعء", "یئ", "وؤ"]
_SOUND_OF = {ch: i for i, grp in enumerate(_SOUND_GROUPS) for ch in grp}
_CANON = {ch: grp[0] for grp in _SOUND_GROUPS for ch in grp}


def _compact(text: str) -> str:
    """Normalise letters and drop spaces / half-spaces so 'کلاب هاوس' == 'کلاب‌هاوس'."""
    return text.translate(_CHAR_MAP).replace(" ", "").replace("‌", "").lower()


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    ga = _SOUND_OF.get(a)
    return 0.5 if ga is not None and ga == _SOUND_OF.get(b) else 1.0


def _word_costs(heard: str, kw: str, word_of, allow):
    """
    Align `heard` with keyword `kw` (both compact) using an edit distance where swapping
    two same-sounding letters costs 0.5, and return the edit cost charged to each word of
    the keyword. `word_of[j]` is the keyword word that character j belongs to; `allow`
    is each word's allowance (an extra heard letter between two words is charged to the
    word with more allowance).
    """
    n, m = len(heard), len(kw)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = float(i)
    for j in range(1, m + 1):
        d[0][j] = float(j)
    for i in range(1, n + 1):
        a, row, prev = heard[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + _sub_cost(a, kw[j - 1]))

    costs = [0.0] * len(allow)

    def owner_of_gap(j):  # word charged for an extra heard letter before kw[j]
        left = word_of[j - 1] if j > 0 else None
        right = word_of[j] if j < m else None
        if left is None or right is None:
            return left if right is None else right
        return left if allow[left] >= allow[right] else right

    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + _sub_cost(heard[i - 1], kw[j - 1]):
            costs[word_of[j - 1]] += _sub_cost(heard[i - 1], kw[j - 1])
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:   # keyword letter missing
            costs[word_of[j - 1]] += 1
            j -= 1
        else:                                        # extra heard letter
            costs[owner_of_gap(j)] += 1
            i -= 1
    return costs


def _prefix_cost(word: str, text: str, lo: int, hi: int) -> float:
    """Smallest edit cost between `word` and a prefix of `text` of length lo..hi."""
    prev = [float(j) for j in range(len(text) + 1)]
    for i, a in enumerate(word, 1):
        cur = [float(i)]
        for j, b in enumerate(text, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + _sub_cost(a, b)))
        prev = cur
    return min(prev[max(0, lo):hi + 1], default=float("inf"))


def _word_allowance(length: int, fuzziness: float) -> float:
    """How many edits one word of a keyword may have and still count as heard."""
    if length <= 1:
        base = 1.0      # و / ی: dropped or added often
    elif length <= 4:
        base = 0.5      # one same-sounding letter (سفر/صفر), never a different word (علم/عقل)
    elif length <= 7:
        base = 1.0
    else:
        base = 2.0
    return base * fuzziness


def _single_allowance(length: int, fuzziness: float) -> float:
    """Allowance for a one-word keyword (stricter for short words, which collide easily)."""
    if length < 4:
        return 0.0
    return _word_allowance(length, fuzziness)


KEYWORDS_DIR = SCRIPT_DIR / "keywords"   # one .txt keyword list per topic


def keyword_lists(folder: Path = KEYWORDS_DIR):
    """{list name: path} of the keyword lists (keywords/*.txt), setting the folder up first."""
    folder.mkdir(parents=True, exist_ok=True)
    if not any(folder.glob("*.txt")):
        old = SCRIPT_DIR / "keywords.txt"          # a single list from an earlier version
        example = SCRIPT_DIR / "keywords.example.txt"
        if old.is_file():
            shutil.move(str(old), str(folder / "philosophy.txt"))
            print(f"Moved keywords.txt to {folder / 'philosophy.txt'} (one list per topic now)")
        elif example.is_file():
            shutil.copyfile(example, folder / "general.txt")
    return {p.stem: p for p in sorted(folder.glob("*.txt"), key=lambda p: p.stem.lower())}


def count_entries(path: Path) -> int:
    try:
        return sum(1 for raw in path.read_text(encoding="utf-8-sig").splitlines()
                   if raw.split("#", 1)[0].strip())
    except OSError:
        return 0


class KeywordFixer:
    """
    keywords.txt format (UTF-8, one entry per line, '#' starts a comment):
        کلاب‌هاوس                 keyword: close misspellings are snapped to it
        کلاب هوس => کلاب‌هاوس     rule: always replace the left side with the right side
    """

    def __init__(self, path, fuzziness: float = 1.0):
        """path: one keyword list, or several (their entries are combined)."""
        self.fuzziness = fuzziness
        self.rules = []     # (n_words, compact_source, replacement)
        self.keywords = []  # (n_words, compact_form, display_text)
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        seen = set()
        for p in paths:
            p = Path(p)
            if not p.is_file():
                continue
            for raw in p.read_text(encoding="utf-8-sig").splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line or line in seen:
                    continue
                seen.add(line)
                if "=>" in line:
                    src, dst = (s.strip() for s in line.split("=>", 1))
                    if src and dst:
                        self.rules.append((len(src.split()), _compact(src), dst))
                else:
                    self.keywords.append((len(line.split()), _compact(line), line))
        self.rules.sort(key=lambda r: -r[0])  # longest rules first

        # Index keywords by (word count, compact length) so each heard window is only
        # compared with keywords of a similar shape.
        # Index keywords by (first letter, word count); same-sounding letters count as one.
        self._index = {}
        self._kw_set = set()
        for k, kw, display in self.keywords:
            parts = [_compact(p) for p in display.split()]
            parts = [p for p in parts if p]
            if not kw or not parts:
                continue
            word_of = [w for w, p in enumerate(parts) for _ in p]
            if k == 1:
                allow = [_single_allowance(len(kw), fuzziness)]
            else:
                allow = [_word_allowance(len(p), fuzziness) for p in parts]
            entry = (kw, display, word_of, allow, sum(allow), parts[0], allow[0])
            self._index.setdefault((_CANON.get(kw[0], kw[0]), k), []).append(entry)
            self._kw_set.add(kw)
        self._max_words = max((k for k, _, _ in self.keywords), default=0) + 1
        self._cache = {}
        self._first_cache = {}

    def __len__(self):
        return len(self.rules) + len(self.keywords)

    def _cost(self, window: str, kw: str, word_of, allow):
        """Total edit cost if every keyword word is within its allowance, else None."""
        key = (window, kw)
        if key not in self._cache:
            costs = _word_costs(window, kw, word_of, allow)
            ok = all(c <= a for c, a in zip(costs, allow))
            self._cache[key] = sum(costs) if ok else None
        return self._cache[key]

    def _first_word_ok(self, first, a0, window):
        """Cheap pre-check: the keyword's first word must match the start of the window."""
        slack = int(a0) + 1
        key = (first, a0, window[:len(first) + slack])
        if key not in self._first_cache:
            self._first_cache[key] = _prefix_cost(
                first, key[2], len(first) - slack, len(first) + slack) <= a0
        return self._first_cache[key]

    def _is_inflected(self, window):
        """Heard text is some keyword plus a short prefix/suffix (تهرانی, وجود رابطه)."""
        for cut in (1, 2, 3):
            if len(window) - cut >= 3 and (window[:-cut] in self._kw_set or window[cut:] in self._kw_set):
                return True
        return False

    def _best_keyword(self, compact, raw, i):
        if not compact[i]:
            return None
        best = None  # (cost, |n - k|, display, n)
        first_letter = _CANON.get(compact[i][0], compact[i][0])
        window = ""
        for n in range(1, min(self._max_words, len(compact) - i) + 1):
            window += compact[i + n - 1]
            if window in self._kw_set:
                inflected = False
            else:
                inflected = None  # computed on demand
            for k in (n - 1, n, n + 1):
                if k < 1:
                    continue
                for kw, display, word_of, allow, slack, first, a0 in self._index.get((first_letter, k), ()):
                    if abs(len(window) - len(kw)) > slack:
                        continue
                    if window != kw:
                        # A window with an extra word must not start or end on a tiny word
                        # (و / به / از ...) that merely sits next to the keyword.
                        if n > k and (len(raw[i]) <= 2 or len(raw[i + n - 1]) <= 2):
                            continue
                        # Keyword + prefix/suffix is an inflected form, not a mistake.
                        if (window.startswith(kw) or kw.startswith(window)
                                or window.endswith(kw) or kw.endswith(window)):
                            continue
                        if inflected is None:
                            inflected = self._is_inflected(window)
                        if inflected:
                            continue
                        if not self._first_word_ok(first, a0, window):
                            continue
                    cost = self._cost(window, kw, word_of, allow)
                    if cost is not None:
                        cand = (cost, abs(n - k), display, n)
                        if best is None or cand[:2] < best[:2]:
                            best = cand
        return best

    def fix(self, words):
        """words: [(word, start_sec)] -> (fixed words, number of replacements)."""
        out, fixes, i = [], 0, 0
        raw = [w[0] for w in words]
        compact = [_compact(w) for w in raw]
        while i < len(words):
            match = None  # (replacement, n_words)

            # 1. Explicit rules: exact match (ignoring spacing / letter forms).
            for n, src, dst in self.rules:
                if i + n <= len(words) and "".join(compact[i:i + n]) == src:
                    match = (dst, n)
                    break

            # 2. Keywords: the closest one within each word's allowance. The heard text may
            #    have one word more or fewer than the keyword (model split or merged it).
            if match is None and self.keywords:
                best = self._best_keyword(compact, raw, i)
                if best is not None:
                    match = (best[2], best[3])

            if match is None:
                out.append(words[i])
                i += 1
                continue
            text, n = match
            if " ".join(raw[i:i + n]) != text:
                fixes += 1
            out.append((text, words[i][1], 1.0)[:len(words[i])])  # a keyword fix counts as certain
            i += n
        return out, fixes


def make_cues(words, seg_end: float):
    """Group words into cues that fit MAX_LINES x MAX_LINE_CHARS and MAX_CUE_SECONDS."""
    cues, cur = [], []
    limit = MAX_LINE_CHARS * MAX_LINES
    for i, (w, t) in enumerate(words):
        if cur:
            text_len = len(" ".join(x for x, _ in cur)) + 1 + len(w)
            if text_len > limit or (t - cur[0][1]) > MAX_CUE_SECONDS:
                cues.append(cur)
                cur = []
        cur.append((w, t))
    if cur:
        # Don't leave a 1-2 word orphan on its own; tack it onto the previous cue.
        if cues and len(" ".join(w for w, _ in cur)) < 15:
            cues[-1].extend(cur)
        else:
            cues.append(cur)

    out = []
    for i, cue in enumerate(cues):
        start = cue[0][1]
        end = cues[i + 1][0][1] if i + 1 < len(cues) else seg_end
        end = max(end, start + 0.6)
        out.append([start, end, " ".join(w for w, _ in cue)])
    return out


def wrap_lines(text: str) -> str:
    if len(text) <= MAX_LINE_CHARS:
        return text
    words, lines, line = text.split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > MAX_LINE_CHARS and len(lines) < MAX_LINES - 1:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return "\n".join(lines)


def fmt_clock(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def srt_time(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_file(path: Path, work, sherpa_onnx, recognizer, vad_config, itn_for=None,
                    keyword_fixer=None, reviewer=None):
    """work: workdir.WorkDir for this file (saved progress, and where the .srt goes)."""
    t0 = time.time()
    out_path = work.srt
    if reviewer:   # the review page says what is happening while the audio is read
        def reading(frac):
            reviewer.status(f"Opening {path.name}: reading the audio… {frac:.0%}", frac)
        reading(0.0)
    audio = load_audio(path, reading if reviewer else None)
    duration = len(audio) / SAMPLE_RATE
    total = fmt_clock(duration)

    # Lines an earlier run already transcribed (it may have been stopped half-way).
    cached, done = work.load_segments()
    if cached is None:
        print("  the saved progress is for a different version of this file or other settings; starting over")
        work.reset()
        cached, done = [], False
    resume_at = cached[-1][0] + cached[-1][1] if cached else 0.0
    if cached:
        print(f"  continuing: {fmt_clock(duration if done else resume_at)} of {total} "
              f"already transcribed ({len(cached)} lines)")

    decoded = []  # (segment start, segment duration, [(word, time, confidence)])
    if reviewer:  # the review page shows each line as soon as it is transcribed
        reviewer.start_file(path.name, duration, audio, work, cached)  # restores saved edits too
    else:
        decoded.extend(cached)

    if not done:
        work.open_segments(resume=bool(cached))
        last_print = 0.0
        FG_IDLE.clear()     # the background queue waits meanwhile
        try:
            # Speech detection restarts just after the last saved line (a pause, since lines
            # end at one), so nothing is transcribed twice.
            for start, samples in vad_segments(sherpa_onnx, vad_config, audio, resume_at):
                result = recognizer.decode(samples)
                seg_dur = len(samples) / SAMPLE_RATE
                if result.text:
                    words = words_with_times(result, start, seg_dur, result.confidences)
                    work.add_segment(start, seg_dur, words)
                    if reviewer:
                        reviewer.add_segment(start, seg_dur, words)
                    else:
                        decoded.append((start, seg_dur, words))
                if reviewer:
                    reviewer.check_switch()   # another file opened on the page: stop here
                if time.time() - last_print > 0.5:
                    last_print = time.time()
                    print(f"\r  transcribed {fmt_clock(start + seg_dur)} of {total}", end="", flush=True)
                    if reviewer:
                        reviewer.progress(start + seg_dur)
            work.finish_segments()
        finally:
            work.close()
            FG_IDLE.set()
        print(f"\r  transcribed {total} of {total}  ")

    fixes = 0
    if reviewer:  # applies the keywords itself (they may change while reviewing)
        decoded, fixes = reviewer.finish_file()
    elif keyword_fixer:
        fixed = []
        for start, seg_dur, words in decoded:
            words, n = keyword_fixer.fix(words)
            fixes += n
            fixed.append((start, seg_dur, words))
        decoded = fixed

    itn = itn_for() if itn_for else None   # spoken numbers as digits, if chosen (read now)
    n_cues = write_srt(out_path, decoded, itn)

    # Off-topic stretches marked on the review page: also write the lesson without them.
    marks = reviewer.marks if reviewer else []
    stem = out_path.with_suffix("")
    if marks:
        write_lesson(stem, decoded, marks, itn)
        print(f"  {len(marks)} off-topic stretch(es) marked -> {stem.name}.lesson.srt, "
              f"{stem.name}.lesson.txt, {stem.name}.offtopic.csv")
    elif reviewer and reviewer.mode == "ask":   # all marks were removed: drop the old files
        for ext in LESSON_FILES:
            stem.with_name(stem.name + ext).unlink(missing_ok=True)

    # The transcript to read: Markdown and Word, with speakers' names (if found on the page)
    import export
    sp = work.load_json("speakers.json") or {}
    docs = export.write_documents(stem, path.stem, decoded, marks,
                                  [tuple(t) for t in sp.get("turns", [])], sp.get("names"), itn)
    print(f"  also as a document: {', '.join(d.name for d in docs)}")

    took = time.time() - t0
    if keyword_fixer or (reviewer and reviewer.fixer):
        print(f"  keywords: {fixes} correction(s)")
    if reviewer:
        print(reviewer.summary())
        reviewer.file_saved(out_path.name)
    print(f"  {n_cues} subtitles, {duration / 60:.1f} min audio in {took:.0f}s -> {out_path}")


def write_srt(path: Path, decoded, itn=None) -> int:
    """decoded: [(start, dur, [(word, time, ...)])] -> .srt file; returns the number of cues."""
    cues = []
    for start, seg_dur, words in decoded:
        if words:
            cues.extend(make_cues([(w[0], w[1]) for w in words], start + seg_dur))

    # Keep cues in order and stop them from overlapping.
    cues.sort(key=lambda c: c[0])
    for i in range(len(cues) - 1):
        cues[i][1] = min(cues[i][1], cues[i + 1][0] - 0.01)

    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        for i, (start, end, text) in enumerate(cues, 1):
            if itn:
                text = itn(text)
            f.write(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{wrap_lines(text)}\n\n")
    return len(cues)


def write_lesson(stem: Path, decoded, marks, itn=None) -> None:
    """
    Files without the off-topic stretches (marks: [{"a", "b"}], by word start time):
      <stem>.lesson.srt   subtitles of the lesson only (original timings)
      <stem>.lesson.txt   the lesson as plain text, one paragraph per stretch of speech
      <stem>.offtopic.csv start, end and words of each marked stretch
    """
    import csv

    def marked(t):
        return next((m for m in marks if m["a"] <= t < m["b"]), None)

    # Runs of kept words; a run that stops at a marked word ends where that word starts,
    # so its subtitle doesn't stay on screen over the part that was left out.
    runs, removed = [], {id(m): [] for m in marks}   # runs: (start, dur, words, after a cut?)
    cut = False
    for start, dur, words in decoded:
        keep = []
        for w in words:
            m = marked(w[1])
            if m is None:
                keep.append(w)
                continue
            removed[id(m)].append(w[0])
            if keep:
                runs.append((start, w[1] - start, keep, cut))
                keep = []
            cut = True
        if keep:
            runs.append((start, dur, keep, cut))
            cut = False
    lesson = [(s, d, w) for s, d, w, _ in runs]
    write_srt(stem.with_name(stem.name + ".lesson.srt"), lesson, itn)

    # Plain text: a new paragraph where an off-topic stretch was cut out or after a long pause.
    paragraphs, cur, last_end = [], [], None
    for start, dur, words, after_cut in runs:
        cut = last_end is not None and (after_cut or start - last_end > 8)
        if cut and cur:
            paragraphs.append(" ".join(cur))
            cur = []
        text = " ".join(w[0] for w in words)
        cur.append(itn(text) if itn else text)
        last_end = start + dur
    if cur:
        paragraphs.append(" ".join(cur))
    stem.with_name(stem.name + ".lesson.txt").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8-sig")

    with open(stem.with_name(stem.name + ".offtopic.csv"), "w", encoding="utf-8-sig", newline="") as f:
        out = csv.writer(f)
        out.writerow(["start", "end", "start_seconds", "end_seconds", "text"])
        for m in marks:
            out.writerow([srt_time(m["a"]).replace(",", "."), srt_time(m["b"]).replace(",", "."),
                          round(m["a"], 3), round(m["b"], 3), " ".join(removed[id(m)])])


# Set while no recording is being transcribed for the review page: the background queue
# only works then, so the recording you have open always comes first.
FG_IDLE = threading.Event()
FG_IDLE.set()


class BatchWorker:
    """
    The queue: transcribes recordings in the background (saving each line, like the review
    page does), so they are ready to review later. Started from the file list on the page
    ("Transcribe all here"). It pauses while the open recording is being transcribed, and
    lets go of a recording as soon as it is opened on the page (what it did is kept).
    """

    def __init__(self, sherpa_onnx, recognizer, vad_config, settings, work_root, report):
        self.sherpa_onnx, self.recognizer, self.vad_config = sherpa_onnx, recognizer, vad_config
        self.settings, self.work_root, self.report = settings, work_root, report
        self.items = []          # recordings waiting (resolved paths)
        self.current = None      # the one being transcribed
        self.done_sec = 0.0
        self.duration = 0.0
        self.finished = 0        # transcribed since the queue was started
        self.failed = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.skip = threading.Event()     # let go of the current recording
        self.idle = threading.Event()     # no recording in hand
        self.idle.set()
        threading.Thread(target=self._run, daemon=True).start()

    def add(self, paths) -> int:
        n = 0
        with self.lock:
            busy = {str(p) for p in self.items} | ({str(self.current)} if self.current else set())
            for p in paths:
                p = Path(p).resolve()
                if str(p) in busy or workdir.status(p, self.work_root)["state"] in ("transcribed", "done"):
                    continue
                self.items.append(p)
                busy.add(str(p))
                n += 1
            if n and self.current is None and not self.items[:-n]:
                self.finished, self.failed = 0, []
        self.wake.set()
        self._report()
        return n

    def stop(self):
        with self.lock:
            self.items.clear()
            if self.current is not None:
                self.skip.set()
        self._report()

    def release(self, path: Path):
        """The page opens `path`: stop transcribing it here (lines done are kept) and unqueue it."""
        key = str(Path(path).resolve())
        with self.lock:
            self.items = [p for p in self.items if str(p) != key]
            busy = self.current is not None and str(self.current) == key
            if busy:
                self.skip.set()
        if busy:
            self.idle.wait(60)
        self._report()

    def queued(self, path) -> str:
        """"now", "waiting" or "" for a recording (for the file list)."""
        key = str(Path(path).resolve())
        with self.lock:
            if self.current is not None and str(self.current) == key:
                return "now"
            return "waiting" if any(str(p) == key for p in self.items) else ""

    def state(self):
        with self.lock:
            if self.current is None and not self.items and not self.finished and not self.failed:
                return None
            return {"current": self.current.name if self.current else None,
                    "done_sec": round(self.done_sec, 1), "duration": round(self.duration, 1),
                    "waiting": [p.name for p in self.items], "finished": self.finished,
                    "failed": list(self.failed), "paused": self.current is not None and not FG_IDLE.is_set()}

    def _report(self):
        try:
            self.report(self.state())
        except Exception:
            pass

    def _run(self):
        while True:
            self.wake.wait()
            with self.lock:
                if not self.items:
                    self.wake.clear()
                    continue
                path = self.items.pop(0)
                self.current, self.done_sec, self.duration = path, 0.0, 0.0
                self.skip.clear()
                self.idle.clear()
            self._report()
            try:
                if self._transcribe(path):
                    with self.lock:
                        self.finished += 1
            except Exception as e:
                print(f"\n  queue: {path.name} failed: {e}")
                with self.lock:
                    self.failed.append(path.name)
            finally:
                with self.lock:
                    self.current = None
                    self.idle.set()
                self._report()

    def _transcribe(self, path: Path) -> bool:
        work = workdir.WorkDir(workdir.folder_for(path, self.work_root), path, self.settings)
        cached, done = work.load_segments()
        if cached is None:
            work.reset()
            cached, done = [], False
        if done:
            return True
        print(f"\n  queue: transcribing {path.name} in the background")
        audio = load_audio(path)
        if self.skip.is_set():
            return False
        with self.lock:
            self.duration = len(audio) / SAMPLE_RATE
        resume_at = cached[-1][0] + cached[-1][1] if cached else 0.0
        work.open_segments(resume=bool(cached))
        last = 0.0
        try:
            for start, samples in vad_segments(self.sherpa_onnx, self.vad_config, audio, resume_at):
                while not FG_IDLE.is_set() and not self.skip.is_set():
                    if time.time() - last > 2:
                        last = time.time()
                        self._report()
                    time.sleep(0.3)
                if self.skip.is_set():
                    return False
                result = self.recognizer.decode(samples)
                seg_dur = len(samples) / SAMPLE_RATE
                if result.text:
                    work.add_segment(start, seg_dur, words_with_times(result, start, seg_dur, result.confidences))
                with self.lock:
                    self.done_sec = start + seg_dur
                if time.time() - last > 1:
                    last = time.time()
                    self._report()
            work.finish_segments()
        finally:
            work.close()
        print(f"\n  queue: {path.name} transcribed")
        return True


def collect_inputs(items):
    files = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in MEDIA_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"Not found: {item}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Goosh-o-Ghalam: Persian speech -> .srt subtitles and transcripts "
                                             "(Shenava Koochik, offline)")
    ap.add_argument("inputs", nargs="*",
                    help="audio/video files and/or folders (with --review they're optional: "
                         "files can be picked on the page)")
    ap.add_argument("--work-dir",
                    help="where each recording's folder (saved progress and the .srt) is kept "
                         "(default: the 'transcripts' folder next to this program)")
    ap.add_argument("--out-dir", help="also copy each finished .srt into this folder")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="CPU threads for the model")
    ap.add_argument("--max-segment", type=float, default=15.0,
                    help="longest speech chunk sent to the model, in seconds (default 15)")
    ap.add_argument("--digits", action="store_true",
                    help="convert spoken numbers to Persian digits (may also turn 'نه' = 'no' into ۹)")
    ap.add_argument("--keywords",
                    help="keyword lists to use, comma-separated: names of lists in the keywords "
                         "folder (e.g. philosophy) or paths to .txt files; 'none' for no list "
                         "(default: the ones used last; on the review page they're chosen per "
                         "recording)")
    ap.add_argument("--fuzziness", type=float, default=1.0,
                    help="how loosely heard words are matched to keywords: 0 = exact only, 1 = default, 2 = loose")
    ap.add_argument("--review", action="store_true",
                    help="play each part the model is unsure about and ask what was said")
    ap.add_argument("--blank-uncertain", action="store_true",
                    help="write [...] for unsure parts instead of guessing (with --review: for the "
                         "parts left after you stop asking)")
    ap.add_argument("--confidence", type=float, default=None,
                    help="words the main model is less sure of than this (0-1) count as unsure "
                         "and are checked by the other models and highlighted (default 0.65, or "
                         "the last setting used on the review page)")
    ap.add_argument("--review-in-terminal", action="store_true",
                    help="with --review: ask in this window instead of a browser page "
                         "(the Windows console may show Persian letters unjoined)")
    ap.add_argument("--auto-replace", default=None,
                    help="with --review: when Nava is at least this sure (0-1) of other words for an "
                         "unsure part, replace them automatically (or confirm them if it heard the "
                         "same); also when Nava and Vosk agree, both at least 0.5 sure. "
                         "'off' to never do it (default 0.6, or the last setting used)")
    ap.add_argument("--alternatives", default="nava,vosk",
                    help="with --review: other models whose guesses are shown for each unsure part "
                         "(nava, vosk, both comma-separated, or none; default nava,vosk). "
                         "Downloaded on first use: Nava 1.6 GB, Vosk 53 MB")
    ap.add_argument("--overwrite", action="store_true",
                    help="make the .srt again even if it exists (saved progress is still reused)")
    ap.add_argument("--fresh", action="store_true",
                    help="forget saved progress (transcribed lines, review edits, other models' "
                         "guesses) and start the file over")
    ap.add_argument("--model-dir", default=str(SCRIPT_DIR / "models"))
    args = ap.parse_args()

    files = collect_inputs(args.inputs)
    browsing = args.review and not args.review_in_terminal
    if not files and not (browsing and all(Path(i).is_dir() for i in args.inputs)):
        sys.exit("No audio/video files to transcribe." if args.inputs else
                 "Give audio/video files or folders (or use --review to pick them on a page).")

    model_dir = Path(args.model_dir)
    on_page = args.review and not args.review_in_terminal
    if not on_page:      # (with the review page, models are downloaded there, with progress)
        download_models(model_dir)

    # Settings per recording: keyword lists, thresholds, second opinions. New
    # recordings start from the last ones used (or these built-in ones); options given on
    # the command line override them.
    work_root = Path(args.work_dir) if args.work_dir else workdir.DEFAULT_ROOT
    lists = keyword_lists()
    defaults = workdir.clean_settings(workdir.load_defaults(work_root), {
        "keywords": list(lists), "confidence": 0.65, "auto_replace": 0.6,
        "alternatives": True})
    if args.keywords is not None:
        chosen = [k.strip() for k in args.keywords.split(",") if k.strip()]
        if chosen == ["none"]:
            chosen = []
        for k in chosen:
            if k not in lists and not Path(k).is_file():
                sys.exit(f"No keyword list '{k}' (lists: {', '.join(lists) or 'none'}; or give a .txt path)")
        defaults["keywords"] = chosen
    if args.confidence is not None:
        defaults["confidence"] = args.confidence
    if args.auto_replace is not None:
        if args.auto_replace.strip().lower() in ("off", "none", "no", "0"):
            defaults["auto_replace"] = None
        else:
            try:
                defaults["auto_replace"] = float(args.auto_replace)
            except ValueError:
                sys.exit(f"--auto-replace takes a number between 0 and 1, or off (got {args.auto_replace})")
            if not 0 < defaults["auto_replace"] <= 1:
                sys.exit("--auto-replace takes a number between 0 and 1, or off")

    def list_paths(names):
        current = keyword_lists()
        return [current[n] if n in current else Path(n) for n in names
                if n in current or Path(n).is_file()]

    def make_fixer(names):
        paths = list_paths(names)
        fixer = KeywordFixer(paths, args.fuzziness) if paths else None
        return fixer if fixer and len(fixer) else None

    def additions_list(names):
        """Where corrections added while reviewing go: the first chosen list."""
        paths = list_paths(names)
        return paths[0] if paths else KEYWORDS_DIR / "general.txt"

    itn_cache = []

    def get_itn():
        if not itn_cache:
            sys.path.insert(0, str(model_dir))
            from persian_itn import itn as itn_fn
            itn_cache.append(itn_fn)
        return itn_cache[0]

    keyword_fixer = make_fixer(defaults["keywords"])
    kw_path = additions_list(defaults["keywords"])
    if keyword_fixer:
        print(f"Using {len(keyword_fixer)} keyword(s) from {', '.join(defaults['keywords'])}")

    reviewer = web = None
    if args.review or args.blank_uncertain:
        import review
        ui = None
        if args.review:
            if args.review_in_terminal:
                ui = review.TerminalUI()
            else:
                alts = None
                names = [n.strip().lower() for n in args.alternatives.split(",")
                         if n.strip() and n.strip().lower() != "none"]
                if names:
                    try:
                        import alternatives
                        bad = [n for n in names if n not in alternatives.MODELS]
                        if bad:
                            sys.exit(f"Unknown --alternatives model(s): {', '.join(bad)} "
                                     f"(choose from {', '.join(alternatives.MODELS)})")
                        alts = alternatives.AltEngine(model_dir, names,
                                                      threads=max(2, (os.cpu_count() or 4) - 2))
                    except ImportError as e:
                        print(f"Other models' guesses are off ({e}); run setup.bat to install them.")
                ui = web = review.WebUI(alts=alts, media_exts=MEDIA_EXTS, work_root=work_root,
                                        model=(MODEL_TITLE, MODEL_ABOUT, MODEL_URL),
                                        auto_replace=defaults["auto_replace"], model_dir=model_dir,
                                        threads=max(2, (os.cpu_count() or 4) - 2))
        reviewer = review.Reviewer(
            mode="ask" if args.review else "blank",
            threshold=defaults["confidence"],
            kw_path=kw_path,
            fixer=keyword_fixer,
            fixer_factory=lambda p: KeywordFixer(p, args.fuzziness),
            blank_on_stop=args.blank_uncertain,
            ui=ui,
        )

    if on_page and web is not None and models_missing(model_dir):
        # A new computer: the page shows the downloads (the main model is needed to start)
        try:
            download_models(model_dir, status=web.status)
        except Exception as e:
            print(f"Download failed: {e}")
            web.close(f"Setup could not download the speech-recognition model ({e}). "
                      "Check the internet connection, then start review.bat again.")
            sys.exit(1)
    elif on_page:
        download_models(model_dir)
    if web is not None and web.alts is not None:
        web.alts.warm_up()   # Nava and Vosk: download (first time) and load in the background

    if web is not None:
        web.status(f"Loading {MODEL_TITLE}…")
    print(f"Loading model {MODEL_TITLE} ({MODEL_ABOUT}) ...")
    sherpa_onnx, recognizer, vad_config = build_models(model_dir, args.threads, args.max_segment)

    settings = {"model": MODEL_NAME, "max_segment": args.max_segment}
    out_dir = Path(args.out_dir) if args.out_dir else None
    cli_files = {str(p.resolve()) for p in files}

    current = {}   # the recording being worked on: its settings and folder

    def settings_for(path: Path):
        """(settings for a recording, whether they were saved for it before)."""
        work = workdir.WorkDir(workdir.folder_for(path, work_root), path, settings)
        saved = work.load_settings()
        return (workdir.clean_settings(saved, defaults) if saved else dict(defaults)), bool(saved)

    def apply_settings(fs):
        """Use these settings for the recording that is open (called by the page)."""
        fs = workdir.clean_settings(fs, current.get("fs", defaults))
        current["fs"].clear()
        current["fs"].update(fs)
        if current.get("work"):
            current["work"].save_settings(fs)
        workdir.save_defaults(work_root, fs)
        defaults.update(fs)
        fixer = make_fixer(fs["keywords"])
        reviewer.fixer = fixer
        reviewer.kw_path = additions_list(fs["keywords"])
        reviewer.threshold = fs["confidence"]
        return fixer, reviewer.kw_path

    def process(path: Path, label: str, always: bool, from_scratch: bool = False,
                chosen=None) -> bool:
        """Transcribe (and review) one file; False if skipped. chosen: settings from the page."""
        if web is not None and web.batch is not None:
            web.batch.release(path)        # the background queue lets go of it first
        workdir.adopt_old_folder(path, work_root)
        work = workdir.WorkDir(workdir.folder_for(path, work_root), path, settings)
        print(f"{label} {path.name}  ->  {work.dir}")
        if from_scratch:   # "Start over" on the page: everything for this file goes
            work.reset(outputs=True)
            print("  starting over: saved transcript, edits, marks and other models' guesses removed")
        elif args.fresh and str(path.resolve()) in cli_files:
            work.reset()
        if work.srt.exists() and not (args.overwrite or always):
            print("  .srt already exists, skipping (use --overwrite to make it again)")
            return False
        work.create()
        if chosen is not None:              # picked on the page just now
            fs = workdir.clean_settings(chosen, defaults)
            workdir.save_defaults(work_root, fs)
            defaults.update(fs)
        else:
            fs = settings_for(path)[0]
        work.save_settings(fs)
        current.clear()
        current.update(fs=fs, work=work)
        fixer = make_fixer(fs["keywords"])
        print(f"  keyword lists: {', '.join(fs['keywords']) or 'none'}; unsure below "
              f"{fs['confidence']:.0%}" + (f"; auto-replace at {fs['auto_replace']:.0%}"
                                          if web and fs['alternatives'] and fs['auto_replace'] else ""))
        if reviewer:
            reviewer.fixer = fixer
            reviewer.kw_path = additions_list(fs["keywords"])
            reviewer.threshold = fs["confidence"]
            if web is not None:
                web.configure(fs)
        transcribe_file(path, work, sherpa_onnx, recognizer, vad_config,
                        lambda: get_itn() if args.digits else None,
                        fixer, reviewer)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in [work.srt] + [work.srt.with_name(work.srt.stem + e) for e in LESSON_FILES]:
                if f.exists():
                    shutil.copyfile(f, out_dir / f.name)
            print(f"  copied to {out_dir}")
        return True

    if web is not None:
        web.settings_for = lambda p: settings_for(Path(p))
        web.apply_settings = apply_settings
        web.keyword_lists = lambda: {n: count_entries(p) for n, p in keyword_lists().items()}
        web.itn_for = lambda: get_itn() if args.digits else None
        web.batch = BatchWorker(sherpa_onnx, recognizer, vad_config, settings, work_root,
                                lambda st: web._set(queue=st))
        # The page decides what to work on: the files given here first (if any), then
        # whatever is chosen in its file browser, until "Close the program" is pressed.
        queue = list(files)
        start = Path(args.inputs[0]) if len(args.inputs) == 1 and Path(args.inputs[0]).is_dir() else None
        web.set_folder(start or (files[0].parent if files else workdir.last_folder(work_root)))
        nxt = queue.pop(0) if (queue and start is None) else None
        scratch = False   # the next file is to be started over from scratch
        chosen = None     # settings picked for it on the page
        while True:
            if nxt is None:
                action, path = web.wait_for_file(
                    reviewer.saved[-1] if reviewer.saved else "Choose a recording")
                if action == "quit":
                    break
                nxt, scratch, chosen = Path(path), action == "redo", web.picked_settings
                workdir.remember_folder(work_root, nxt.parent)
            try:
                process(nxt, ">>", always=True, from_scratch=scratch, chosen=chosen)
                nxt, scratch, chosen = (queue.pop(0) if queue else None), False, None
            except review.SwitchFile:
                reviewer.abandon_file()
                action, path = web.take_pending()
                if action == "quit":
                    break
                print("\n  " + ("starting this file over" if action == "redo" else
                               "switched to another file; this one is saved and continues when reopened"))
                nxt = Path(path) if action in ("open", "redo") else None
                scratch, chosen = action == "redo", web.picked_settings
                if nxt is not None:
                    workdir.remember_folder(work_root, nxt.parent)
            except Exception as e:  # show it on the page and go back to the file list
                print(f"  FAILED: {e}")
                reviewer.abandon_file()
                reviewer.saved.append(f"Failed on {nxt.name}: {e}")
                nxt = None
        web.batch.stop()                # a recording half-done in the queue carries on next time
        web.batch.idle.wait(15)
        print("Closed.")
        reviewer.close("Closed.")
        # Leave at once: the other models' threads may still be loading in native code, and
        # Python's usual shutdown can then end with an error code (which keeps review.bat's
        # window open). Everything is saved by now (each file is written whole, then renamed).
        sys.stdout.flush()
        os._exit(0)

    failed = 0
    for i, path in enumerate(files, 1):
        try:
            process(path, f"[{i}/{len(files)}]", always=False)
        except Exception as e:  # keep going with the rest of the batch
            failed += 1
            print(f"  FAILED: {e}")

    print("Done." if not failed else f"Done, {failed} file(s) failed.")
    if reviewer:
        reviewer.close("All done." if not failed else f"Done, {failed} file(s) failed.")


if __name__ == "__main__":
    main()
