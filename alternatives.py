"""
Second opinions for the review page: other speech recognisers transcribe the same short
clip the user hears, and the words that line up with the selected part are offered as
suggestions.

  Nava  Whisper large-v3-turbo fine-tuned on Persian (Nimruz/Nava-ct2, via faster-whisper)
  Vosk  Kaldi recogniser with a Persian word list (vosk-model-small-fa-0.42)

Models are downloaded into models/alt on first use and run in one background thread:
clips the user opens go first, then the unsure parts in order, as they are transcribed.
"""

import difflib
import heapq
import itertools
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
NAVA_REPO = "Nimruz/Nava-ct2"
VOSK_NAME = "vosk-model-small-fa-0.42"
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_NAME}.zip"
NOW, SOON, LATER = 0, 1, 2   # opened by the user / the next few after it / the rest
NAVA_WINDOW = 15             # seconds of audio Nava looks at per clip (Whisper's own is 30)

_NORM = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "أ": "ا", "إ": "ا", "ٱ": "ا", "ۀ": "ه",
    "ـ": None, "‌": None, "؟": None, "?": None, ".": None, "،": None, ",": None,
    "!": None, "«": None, "»": None, ":": None, "؛": None,
    **{chr(c): None for c in range(0x064B, 0x0653)},
})


def norm(word: str) -> str:
    return word.translate(_NORM)


def _pieces(words):
    """Split words at half-spaces (می‌کنم ~ می کنم): [(piece, index of its word)]."""
    out = []
    for k, w in enumerate(words):
        for part in w.split("‌"):
            part = norm(part)
            if part:
                out.append((part, k))
    return out


def align(before, span, after, alt_words):
    """
    Which of alt_words correspond to `span`, given our words around it in the same clip?
    Returns (lo, hi): alt_words[lo:hi] (possibly empty), or None if it can't be told.
    """
    found = align_info(before, span, after, alt_words)
    return found[:2] if found else None


AUTO_REPLACE = 0.6    # Nava this sure of other words: replace an unsure part with them
AUTO_AGREE = 0.5      # Nava and Vosk agreeing, both at least this sure: replace too


def auto_decision(before, span, after, results, threshold=AUTO_REPLACE, main_conf=0.0):
    """
    Should an unsure part (`span`, with our words before/after it in the clip) be settled
    by the other models? results: {model name: {"words", "conf"}}; main_conf: how sure the
    main model was of the part. Returns (text, who, confidence) or None; text equal to the
    span means "confirmed".
    Rules: Nava at least `threshold` sure; or Nava and Vosk giving the same words, both at
    least AUTO_AGREE sure. Vosk alone never decides: its scores are too optimistic (it gives
    100% to almost any word in its dictionary, wrong ones included). Replacing also needs the
    other model to be surer than the main model was. The match must be pinned on both
    sides and not much longer than the part.
    """
    picks = {}
    for name, r in results.items():
        if not r or not r.get("conf") or not r.get("words"):
            continue
        found = align_info(before, span, after, r["words"])
        if not found:
            continue
        lo, hi, pin_b, pin_a = found
        words = r["words"][lo:hi]
        if not words or not (pin_b and pin_a) or len(words) > len(span) + 1:
            continue
        picks[name] = (" ".join(words), min(r["conf"][lo:hi]))
    nava, vosk = picks.get("Nava"), picks.get("Vosk")
    decision = None
    if nava and nava[1] >= threshold:
        decision = nava[0], "Nava", nava[1]
    elif (nava and vosk and _same(nava[0], vosk[0])
            and min(nava[1], vosk[1]) >= min(AUTO_AGREE, threshold)):
        decision = nava[0], "Nava and Vosk", min(nava[1], vosk[1])
    # Different words only win if the other model is surer of them than the main model
    # was of its own (confirming the same words needs no such check).
    if decision and not _same(decision[0], " ".join(span)) and decision[2] <= main_conf:
        return None
    return decision


def _same(x, y) -> bool:
    return "".join(p for p, _ in _pieces(x.split())) == "".join(p for p, _ in _pieces(y.split()))


def align_info(before, span, after, alt_words):
    """
    Like align(), plus whether the match is pinned down on each side:
    (lo, hi, anchored before, anchored after), or None. A side with no words of ours in
    the clip counts as pinned (the clip edge pins it).
    """
    ours = _pieces(before + span + after)
    theirs = _pieces(alt_words)
    # near-identical spellings (مفایم ~ مفاهیم) count as the same word
    vocab = {x for x, _ in ours}
    for n, (x, k) in enumerate(theirs):
        if x not in vocab and len(x) >= 3:
            best = max(vocab, key=lambda v: difflib.SequenceMatcher(None, x, v).ratio(), default=None)
            if best and difflib.SequenceMatcher(None, x, best).ratio() >= 0.8:
                theirs[n] = (best, k)
    p, q = len(before), len(before) + len(span)
    lo, hi = 0, len(alt_words)
    anchored = False
    pin_before, pin_after = not before, not after
    sm = difflib.SequenceMatcher(None, [x for x, _ in ours], [x for x, _ in theirs], autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            oi, ti = ours[blk.a + k][1], theirs[blk.b + k][1]
            anchored = True
            if oi < p:
                lo = max(lo, ti + 1)
                pin_before = True     # a word of ours before the part matched
            elif oi >= q:
                hi = min(hi, ti)
                pin_after = True      # a word of ours after it matched
    if (before or after) and not anchored:
        return None  # nothing in common with our words around it: can't tell which part
    return (lo, hi, pin_before, pin_after) if lo <= hi else None


# --------------------------------------------------------------------------- models
class Nava:
    name = "Nava"
    note = "Whisper turbo, Persian fine-tune"
    about = "Whisper large-v3-turbo fine-tuned for Persian · 809M parameters (1.6 GB)"
    url = "https://huggingface.co/Nimruz/Nava"

    def __init__(self, model_dir: Path, threads: int):
        self.dir = model_dir / "alt" / "nava-ct2"
        self.threads = threads
        self.model = None

    def load(self, status):
        if self.model is not None:
            return
        from faster_whisper import WhisperModel
        missing = [f for f in NAVA_FILES if not (self.dir / f).exists()]
        if missing:   # first time on this computer: about 1.6 GB, mostly model.bin
            for f in missing:
                _download(f"https://huggingface.co/{NAVA_REPO}/resolve/main/{f}", self.dir / f,
                          status, "1.6 GB" if f == "model.bin" else None)
        status("loading…")
        self.model = WhisperModel(str(self.dir), device="cpu", compute_type="int8",
                                  cpu_threads=self.threads)

    def transcribe(self, clip: np.ndarray):
        """-> (words, confidences 0..1 or None)."""
        # Whisper normally pads every clip to a 30 s window, and that window is most of the
        # work. A 15 s window (longer only for longer clips) gives the same text in under
        # half the time; much shorter windows start to make things up.
        # Whisper can also loop on short clips («آآآآ…»): repeats are blocked, and the
        # length is capped at roughly what can be said in the clip.
        import ctranslate2
        seconds = len(clip) / SAMPLE_RATE
        max_tokens = int(seconds * 20) + 15
        try:
            fw = self.model
            feats = fw.feature_extractor(clip.astype(np.float32), padding=0)
            n = feats.shape[-1]
            window = min(3000, max(NAVA_WINDOW * 100, n + (n % 2)))
            padded = np.zeros((feats.shape[0], window), dtype=np.float32)
            padded[:, :min(n, window)] = feats[:, :window]
            tok = fw.hf_tokenizer
            prompt = [tok.token_to_id(t) for t in
                      ("<|startoftranscript|>", "<|fa|>", "<|transcribe|>", "<|notimestamps|>")]
            eot = tok.token_to_id("<|endoftext|>")
            res = fw.model.generate(ctranslate2.StorageView.from_array(padded[None]), [prompt],
                                    beam_size=1, max_length=max_tokens,
                                    no_repeat_ngram_size=3, repetition_penalty=1.1,
                                    return_logits_vocab=True)
            ids = res[0].sequences_ids[0]
            try:  # each token's probability, from the scores the model gave every candidate
                logits = res[0].logits[0]
                probs = []
                for k, tid in enumerate(ids):
                    row = np.asarray(logits[k], dtype=np.float64).reshape(-1)
                    row = np.exp(row - row.max())
                    probs.append(float(row[tid] / row.sum()))
            except (AttributeError, IndexError, TypeError, ValueError):
                probs = None
            # Words start at tokens beginning with a space ("Ġ"); a word's text is decoded
            # from all its tokens at once (one Persian letter can span two tokens), and its
            # confidence is that of its least certain token, as for the main model.
            words, confs, cur, cur_p = [], [], [], []
            for k, tid in enumerate(ids):
                if tid >= eot:
                    continue
                if cur and tok.id_to_token(tid).startswith("Ġ"):
                    words.append(tok.decode(cur).strip())
                    confs.append(min(cur_p) if probs else None)
                    cur, cur_p = [], []
                cur.append(tid)
                if probs:
                    cur_p.append(probs[k])
            if cur:
                words.append(tok.decode(cur).strip())
                confs.append(min(cur_p) if probs else None)
            keep = [k for k, w in enumerate(words) if w]
            return [words[k] for k in keep], ([confs[k] for k in keep] if probs else None)
        except (AttributeError, TypeError):  # a faster-whisper version without these internals
            segments, _ = self.model.transcribe(
                clip.astype(np.float32), language="fa", task="transcribe", beam_size=1,
                without_timestamps=True, condition_on_previous_text=False, vad_filter=False,
                no_repeat_ngram_size=3, repetition_penalty=1.1, max_new_tokens=max_tokens)
            return " ".join(s.text.strip() for s in segments).split(), None


class Vosk:
    name = "Vosk"
    note = "Kaldi, small Persian model"
    about = "Small Kaldi model for Persian, writes only dictionary words · 53 MB"
    url = "https://alphacephei.com/vosk/models"

    def __init__(self, model_dir: Path, threads: int):
        self.dir = model_dir / "alt" / VOSK_NAME
        self.model = None

    def load(self, status):
        if self.model is not None:
            return
        import vosk
        vosk.SetLogLevel(-1)
        if not self.dir.exists():   # first time on this computer: 53 MB
            zip_path = self.dir.parent / (VOSK_NAME + ".zip")
            _download(VOSK_URL, zip_path, status, "53 MB")
            status("unpacking…")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(self.dir.parent)
            zip_path.unlink()
        status("loading…")
        self.model = vosk.Model(str(self.dir))

    def transcribe(self, clip: np.ndarray):
        """-> (words, confidences 0..1): Vosk's own confidence for each word."""
        import vosk
        rec = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.SetWords(True)
        rec.AcceptWaveform((np.clip(clip, -1, 1) * 32767).astype("<i2").tobytes())
        result = json.loads(rec.FinalResult())
        items = [w for w in result.get("result", []) if w.get("word")]
        if not items:
            return result.get("text", "").split(), None
        return [w["word"] for w in items], [float(w.get("conf", 1.0)) for w in items]


MODELS = {"nava": Nava, "vosk": Vosk}
NAVA_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.json", "model.bin")


def _download(url, dest, status, size_label=None):
    """Fetch a model file, reporting "downloading 35% of 1.6 GB…" through status()."""
    import time
    from workdir import fetch
    last = [0.0]

    def progress(done, total):
        if time.time() - last[0] < 0.5 and done != total:
            return
        last[0] = time.time()
        size = size_label or (f"{total / 1e6:.0f} MB" if total else "")
        status(f"downloading {done / total:.0%} of {size}…" if total else f"downloading {done / 1e6:.0f} MB…")

    status(f"downloading{' ' + size_label if size_label else ''}…")
    fetch(url, dest, progress)


def as_result(saved):
    """A saved result: {"words", "conf"}; older saves kept just the text."""
    if isinstance(saved, str):
        return {"words": saved.split(), "conf": None}
    if isinstance(saved, dict) and isinstance(saved.get("words"), list):
        conf = saved.get("conf")
        ok = isinstance(conf, list) and len(conf) == len(saved["words"])
        return {"words": [str(w) for w in saved["words"]], "conf": conf if ok else None}
    return None


# --------------------------------------------------------------------------- engine
class AltEngine:
    """
    Runs each model in its own background thread (so a fast model never waits behind a
    slow one); results are cached per clip. Requests have a priority (NOW, SOON, LATER);
    a LATER request is dropped when it comes up if `wanted(key)` says it's no longer needed
    (the part was already reviewed, or belongs to an earlier file).
    """

    def __init__(self, model_dir: Path, names=("nava", "vosk"), threads: int = 4, wanted=None):
        self.models = [MODELS[n](model_dir, threads) for n in names]
        self.wanted = wanted
        self.status = {m.name: "waiting" for m in self.models}
        self.current = {m.name: None for m in self.models}   # key being worked on
        self.cache = {}            # key -> {model name: {"words": [...], "conf": [...] | None}}
        self.on_result = None      # (key, model name, result) -> saved
        self.hold = None           # -> True while they should wait (a live session is on)
        self.queues = {m.name: ([], {}) for m in self.models}  # heap, key -> best priority
        self.count = itertools.count()
        self.cv = threading.Condition()
        self._load_locks = {m.name: threading.Lock() for m in self.models}
        for m in self.models:
            threading.Thread(target=self._work, args=(m,), daemon=True).start()

    def request(self, key, clip, priority=LATER):
        with self.cv:
            done = self.cache.get(key, {})
            for m in self.models:
                heap, best = self.queues[m.name]
                if m.name in done or best.get(key, 99) <= priority:
                    continue
                best[key] = priority
                heapq.heappush(heap, (priority, next(self.count), key, clip))
            self.cv.notify_all()

    def preload(self, results):
        """Results saved by an earlier run: {key: {model name: result}}."""
        with self.cv:
            for key, got in results.items():
                for m in self.models:
                    r = as_result(got.get(m.name))
                    if r is not None:
                        self.cache.setdefault(key, {})[m.name] = r

    def warm_up(self):
        """Download (first time) and load every model now, in the background."""
        for m in self.models:
            threading.Thread(target=self._load, args=(m,), daemon=True).start()

    def _load(self, m):
        """Load a model once (from whichever thread gets there first); False if it can't."""
        lock = self._load_locks.setdefault(m.name, threading.Lock())
        with lock:
            if m.model is not None:
                return True
            if self.status[m.name].startswith("unavailable"):
                return False
            try:
                m.load(lambda s: self._set_status(m.name, s))
                self._set_status(m.name, "ready")
                return True
            except Exception as e:
                self._set_status(m.name, f"unavailable: {e}")
                return False

    def get(self, key):
        """{model name: {"words", "conf"}} of the models that have finished this clip."""
        with self.cv:
            return dict(self.cache.get(key, {}))

    def results(self, key):
        with self.cv:
            got = self.cache.get(key, {})
            out = []
            for m in self.models:
                st = self.status[m.name]
                if m.name in got:
                    st = "done"
                elif not st.startswith(("unavailable", "downloading", "loading")):
                    st = "listening…" if self.current[m.name] == key else "queued…"
                r = got.get(m.name)
                out.append({"name": m.name, "note": m.note, "about": m.about, "url": m.url, "status": st,
                            "words": r["words"] if r else None, "conf": r["conf"] if r else None})
            return out

    def _set_status(self, name, text):
        with self.cv:
            self.status[name] = text

    def _next(self, m):
        heap, best = self.queues[m.name]
        while True:
            with self.cv:
                while not heap:
                    self.cv.wait()
                priority, _, key, clip = heapq.heappop(heap)
                if best.get(key) != priority:
                    continue                       # superseded by a more urgent request
                del best[key]
                if m.name in self.cache.get(key, {}):
                    continue
            if priority >= LATER and self.wanted is not None and not self.wanted(key):
                continue                           # already reviewed, or an earlier file
            with self.cv:
                self.current[m.name] = key
            return key, clip

    def _work(self, m):
        while True:
            key, clip = self._next(m)
            if not self._load(m):          # a model that can't load is skipped from then on
                with self.cv:
                    self.current[m.name] = None
                continue
            while self.hold is not None and self.hold():
                time.sleep(0.3)
            try:
                words, conf = m.transcribe(clip)
                result = {"words": list(words), "conf": [round(c, 3) for c in conf] if conf else None}
            except Exception:
                result = {"words": [], "conf": None}
            with self.cv:
                self.cache.setdefault(key, {})[m.name] = result
                self.current[m.name] = None
                save = self.on_result
            if save:
                try:
                    save(key, m.name, result)
                except OSError:
                    pass  # saving is a convenience; never stop the model over it
