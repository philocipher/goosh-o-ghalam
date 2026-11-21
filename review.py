"""
Review a transcript before the subtitles are written: words the model is unsure about are
highlighted (or blanked), and any word can be corrected.

Used by transcribe.py with --review (edit) or --blank-uncertain (blank without asking).
Word confidences come from transcribe.Recognizer (each token's probability).

    ReviewSession  one file: its segments, words, edits, and keyword additions
    WebUI          the editor page in the browser (default; browsers show Persian properly)
    TerminalUI     the old question-by-question prompts in the console
    Reviewer       ties them together for each file
"""

import io
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

SAMPLE_RATE = 16000
BLANK = "[…]"
CONTEXT_WORDS = 8        # words of context shown on each side (terminal)
CLIP_PAD = 1.0           # seconds of audio played before/after the selected words
MAX_SPAN_WORDS = 4       # longest run of unsure words treated as one unsure part
MAX_CLIP = 60.0          # longest clip served, seconds
PAGE = Path(__file__).resolve().parent / "review_page.html"
ACTIONS = ("keep", "replace", "blank", "delete")


AUDIO_CHUNK = 1 << 20    # bytes per range response of the full-file audio (~33 s)
REVIEW_PORT = 47811      # first port tried for the review page
# The program closes itself when no review page has asked for news this long (a page in a
# background tab still does, at least once a minute), or soon after the last page says goodbye.
PAGE_GONE = 180
PAGE_BYE = 20


def _wav_header(data_bytes: int) -> bytes:
    import struct
    return (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
            + b"data" + struct.pack("<I", data_bytes))


def wav_bytes(clip: np.ndarray) -> bytes:
    pcm = (np.clip(clip, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _iso(text: str) -> str:
    """Isolate Persian text so it keeps its place inside an English sentence."""
    return f"⁨{text}⁩"


def _fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _same_words(x: str, y: str) -> bool:
    try:
        from alternatives import _same
        return _same(x, y)
    except ImportError:
        return x.split() == y.split()


def rule_allowed(heard: str) -> bool:
    # A rule rewrites its left side everywhere, so a short everyday word (سر, از, یه)
    # must not become one.
    return len(heard.split()) > 1 or len(heard) > 3


# --------------------------------------------------------------------------- session
class ReviewSession:
    """
    One file under review. Each segment keeps the words as recognised ("raw") and the
    current words, each {"w": text, "t": start sec, "c": confidence, "s": state, "u": span}.
      state: "" (untouched) | "ok" (kept after review) | "edited" | "blank"
      span:  set on untouched low-confidence words; consecutive words share it
    Keyword-list additions are re-applied to every segment the user hasn't touched.
    Segments are added while the file is still being transcribed (add_segment); each has a
    revision number that goes up whenever its words change.
    """

    def __init__(self, name, duration, audio, threshold, fixer, kw_path, fixer_factory,
                 decoded=()):
        self.name = name
        self.duration = duration
        self.audio = audio
        self.threshold = threshold
        self.fixer = fixer
        self.kw_path = kw_path
        self.fixer_factory = fixer_factory
        self.stopped = False
        self.transcribing = True
        self.decoded_until = 0.0
        self.on_change = None   # called after every edit (to save them)
        self.on_punct = None    # called when the punctuation changes (to save it)
        self.marks = []         # off-topic stretches: [{"a": start sec, "b": end sec}]
        self.history = []       # undo: [(what, {line id: (words, touched)}, marks)], newest last
        self.place = 0.0        # where the player was last (seconds), to carry on from there
        # Punctuation, kept apart from the words (keyword fixes and edits redo the words):
        # {line id: {word start time "12.345": mark}}; the mark follows the word at that time.
        self.punct = {}
        self.punct_manual = {}  # marks set or removed by hand ("" = none): win over the model's
        self.punct_lines = 0    # lines the model has punctuated (later ones are new)
        self.segments = []
        self.stats = {"unsure": 0, "corrected": 0, "kept": 0, "blanked": 0,
                      "deleted": 0, "added": 0, "auto": 0, "confirmed": 0}
        for start, dur, words in decoded:
            self.add_segment(start, dur, words)

    def add_segment(self, start, dur, words):
        seg = {"id": len(self.segments), "start": start, "end": start + dur, "raw": words,
               "touched": False, "rev": 0}
        self._refix(seg)
        self.segments.append(seg)
        self.stats["unsure"] += len(self.spans(seg))
        self.decoded_until = max(self.decoded_until, start + dur)
        return seg

    # -- words ---------------------------------------------------------------
    def _refix(self, seg):
        words, n = (self.fixer.fix(seg["raw"]) if self.fixer else (seg["raw"], 0))
        seg["fixes"] = n
        seg["words"] = [{"w": w[0], "t": float(w[1]), "c": float(w[2]) if len(w) > 2 else 1.0, "s": ""}
                        for w in words]
        self._mark(seg)

    def _unsure(self, w):
        return w["s"] == "" and w["c"] < self.threshold

    def _apply_punct(self, seg):
        """Put the punctuation marks on the words: each goes after the word it belongs to by time."""
        ws = seg["words"]
        for w in ws:
            w.pop("p", None)
        marks = dict(self.punct.get(seg["id"], {}))
        marks.update(self.punct_manual.get(seg["id"], {}))
        if not ws or not marks:
            return
        for key, mark in marks.items():
            if not mark:
                continue
            t = float(key)
            k = max((n for n, w in enumerate(ws) if w["t"] <= t + 0.002), default=None)
            if k is not None and ws[k]["s"] != "blank":
                ws[k]["p"] = mark

    def set_punct(self, marks, lines):
        """The model's punctuation for the first `lines` lines: {line id: {time: mark}}."""
        self.punct = marks
        self.punct_lines = lines
        for seg in self.segments:
            self._apply_punct(seg)
            seg["rev"] += 1
        if self.on_punct:
            self.on_punct()

    def export_punct(self):
        return {"format": 1, "lines": self.punct_lines,
                "model": {str(k): v for k, v in self.punct.items()},
                "manual": {str(k): v for k, v in self.punct_manual.items()}}

    def restore_punct(self, data):
        if not isinstance(data, dict):
            return
        def read(d):
            out = {}
            for k, v in (d or {}).items():
                try:
                    out[int(k)] = {str(t): str(m) for t, m in v.items()}
                except (ValueError, AttributeError):
                    continue
            return out
        self.punct = read(data.get("model"))
        self.punct_manual = read(data.get("manual"))
        self.punct_lines = int(data.get("lines") or 0) if str(data.get("lines", "0")).isdigit() else 0
        for seg in self.segments:
            self._apply_punct(seg)

    def _mark(self, seg):
        self._apply_punct(seg)
        ws = seg["words"]
        for w in ws:
            w.pop("u", None)
        i = 0
        while i < len(ws):
            if self._unsure(ws[i]):
                j = i + 1
                while j < len(ws) and j - i < MAX_SPAN_WORDS and self._unsure(ws[j]):
                    j += 1
                # Single letters (و / ه) aren't worth a question.
                if len("".join(w["w"] for w in ws[i:j])) >= 2:
                    for w in ws[i:j]:
                        w["u"] = i
                i = j
            else:
                i += 1

    def spans(self, seg):
        """[(i, j)] unsure parts of a segment (not those inside off-topic marks)."""
        out, i, ws = [], 0, seg["words"]
        while i < len(ws):
            if "u" in ws[i]:
                j = i + 1
                while j < len(ws) and ws[j].get("u") == ws[i]["u"]:
                    j += 1
                if not self.is_marked(ws[i]["t"]):
                    out.append((i, j))
                i = j
            else:
                i += 1
        return out

    # -- off-topic marks ---------------------------------------------------------
    # Stored as time ranges [a, b) rather than on words, so they can start and end
    # anywhere, span lines, and survive edits and keyword fixes inside them.
    def is_marked(self, t) -> bool:
        return any(m["a"] <= t < m["b"] for m in self.marks)

    def set_mark(self, a, b, on: bool):
        """Mark [a, b) off-topic, or unmark it; overlapping marks are cut or joined."""
        a, b = float(a), float(b)
        if not (0 <= a < b <= self.duration + 1):
            raise ValueError("bad time range")
        self._remember(set(), f"{'marking' if on else 'unmarking'} {_fmt_time(a)}–{_fmt_time(b)} off-topic")
        out = []
        for m in self.marks:
            if m["b"] <= a or m["a"] >= b:
                out.append(m)
                continue
            if m["a"] < a:
                out.append({"a": m["a"], "b": a})
            if m["b"] > b:
                out.append({"a": b, "b": m["b"]})
        if on:
            out.append({"a": round(a, 3), "b": round(b, 3)})
        out.sort(key=lambda m: m["a"])
        merged = []
        for m in out:   # join touching or overlapping marks
            if merged and m["a"] <= merged[-1]["b"] + 1e-6:
                merged[-1]["b"] = max(merged[-1]["b"], m["b"])
            else:
                merged.append(dict(m))
        self.marks = merged
        if self.on_change:
            self.on_change()

    def left(self) -> int:
        return sum(len(self.spans(s)) for s in self.segments)

    def first_unsure(self):
        for seg in self.segments:
            sp = self.spans(seg)
            if sp:
                return seg, sp[0][0], sp[0][1]
        return None

    def end_of(self, seg, j):
        return seg["words"][j]["t"] if j < len(seg["words"]) else seg["end"]

    def clip(self, a: float, b: float) -> bytes:
        a = max(0.0, a)
        b = min(max(a, b), a + MAX_CLIP)
        return wav_bytes(self.audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)])

    def pcm(self) -> bytes:
        """The whole file as 16-bit PCM (built once, for the page's full player)."""
        if getattr(self, "_pcm", None) is None:
            self._pcm = (np.clip(self.audio, -1, 1) * 32767).astype("<i2").tobytes()
        return self._pcm

    def clip_for(self, seg, i, j) -> bytes:
        return self.clip(seg["words"][i]["t"] - CLIP_PAD, self.end_of(seg, j) + CLIP_PAD)

    def alt_key(self, seg, i, j):
        """(start, end) of the clip the page plays for words[i:j]."""
        a = round(max(0.0, seg["words"][i]["t"] - CLIP_PAD), 2)
        return a, round(min(self.end_of(seg, j) + CLIP_PAD, a + MAX_CLIP), 2)

    def unsure_after(self, seg_id, i, n):
        """The next n unsure parts after words[i] of a segment, as (segment, i, j)."""
        out = []
        for seg in self.segments[seg_id:]:
            for a, b in self.spans(seg):
                if seg["id"] > seg_id or a > i:
                    out.append((seg, a, b))
                    if len(out) == n:
                        return out
        return out

    def alt_request(self, seg, i, j):
        """
        What other models need for words[i:j]: (clip key, samples, words before / span /
        after inside the clip). The clip is the one the page plays.
        """
        ws = seg["words"]
        a, b = self.alt_key(seg, i, j)
        samples = self.audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]
        # the clip can reach into the neighbouring lines: their words are context too
        k = seg["id"]
        prev = self.segments[k - 1]["words"] if k > 0 else []
        nxt = self.segments[k + 1]["words"] if k + 1 < len(self.segments) else []
        before = [w["w"] for w in prev + ws[:i] if w["t"] >= a]
        after = [w["w"] for w in ws[j:] + nxt if w["t"] < b]
        return (a, b), samples, before, [w["w"] for w in ws[i:j]], after

    # -- edits ---------------------------------------------------------------
    def apply_edit(self, seg_id, i, j, action, text=None, add=None):
        """
        Apply one edit to words[i:j] of a segment. Returns
        {"changed": [segment ids], "notice": str | None}.
        """
        if not (isinstance(seg_id, int) and 0 <= seg_id < len(self.segments)):
            raise ValueError("bad segment")
        seg = self.segments[seg_id]
        ws = seg["words"]
        if not (isinstance(i, int) and isinstance(j, int) and 0 <= i < j <= len(ws)):
            raise ValueError("bad word range")
        if action not in ACTIONS:
            raise ValueError("bad action")

        heard = " ".join(w["w"] for w in ws[i:j])
        text = " ".join((text or "").split())[:500]
        if action == "replace" and (not text or text == heard):
            action = "keep"
        self._remember({seg_id}, {"keep": "keeping", "blank": "blanking", "delete": "deleting",
                                  "replace": "correcting"}[action] + f" «{_iso(heard)}»")

        t0, t1 = ws[i]["t"], self.end_of(seg, j)
        if action == "keep":
            new = [dict(w, s=w["s"] or "ok") for w in ws[i:j]]
            self.stats["kept"] += 1
        elif action == "blank":
            new = [{"w": BLANK, "t": t0, "c": 1.0, "s": "blank", "was": self._original(ws[i:j])}]
            self.stats["blanked"] += 1
        elif action == "delete":
            new = []
            self.stats["deleted"] += 1
        else:
            new = self._edited_words(text, t0, t1, ws[i:j])
            self.stats["corrected"] += 1
        seg["words"] = ws[:i] + new + ws[j:]
        seg["touched"] = True
        seg["rev"] += 1
        self._mark(seg)
        changed = [seg_id]

        notice = None
        if action == "replace" and (add == "keyword" or (add == "rule" and rule_allowed(heard))):
            line = text if add == "keyword" else f"{heard} => {text}"
            self._append_keyword(line)
            self.stats["added"] += 1
            changed += self._refix_untouched()
            if add == "keyword":
                notice = f"Added keyword “{_iso(text)}” to {self.kw_path.name}"
            else:
                notice = (f"Added rule to {self.kw_path.name}: "
                          f"“{_iso(heard)}” → “{_iso(text)}” everywhere")
            n = len(changed) - 1
            if n:
                notice += f" (updated {n} other line{'s' if n > 1 else ''})"
        if self.on_change:
            self.on_change()
        return {"changed": changed, "notice": notice}

    @staticmethod
    def _original(words):
        """What the main model heard for these words (they may be an earlier correction)."""
        out = []
        for w in words:
            x = w.get("was") or w["w"]
            if not out or out[-1] != x or "was" not in w:
                out.append(x)
        return " ".join(out)

    def _edited_words(self, text, t0, t1, replaced):
        parts = text.split()
        step = (t1 - t0) / len(parts)
        was = self._original(replaced) if replaced else ""
        return [{"w": p, "t": round(t0 + k * step, 3), "c": 1.0, "s": "edited",
                 **({"was": was} if was and was != text else {})}
                for k, p in enumerate(parts)]

    # -- undo -----------------------------------------------------------------------
    UNDO_STEPS = 100

    def _remember(self, seg_ids, what):
        """Keep the lines (and marks) as they are now, so the next change can be undone."""
        import copy
        snap = {k: (copy.deepcopy(self.segments[k]["words"]), self.segments[k]["touched"])
                for k in seg_ids}
        self.history.append((what, snap, [dict(m) for m in self.marks]))
        del self.history[:-self.UNDO_STEPS]

    def undo_label(self):
        return self.history[-1][0] if self.history else None

    def undo(self):
        """Put back the lines and marks as they were before the last change; -> changed line ids."""
        if not self.history:
            raise ValueError("nothing to undo")
        what, snap, marks = self.history.pop()
        for k, (words, touched) in snap.items():
            seg = self.segments[k]
            seg["words"] = words
            seg["touched"] = touched
            if not touched:
                self._refix(seg)     # keyword lists may have changed since
            seg["rev"] += 1
            self._mark(seg)
        changed = sorted(snap)
        if marks != self.marks:
            self.marks = marks
            for seg in self.segments:
                seg["rev"] += 1
            changed = list(range(len(self.segments)))
        if self.on_change:
            self.on_change()
        return changed, what

    # -- replace all ------------------------------------------------------------------
    def find_words(self, find):
        """Every place the words `find` occur (whole words, spelling variants alike): [(line, i, j)]."""
        from workdir import normalize_fa
        want = normalize_fa(find).split()
        if not want:
            return []
        out = []
        for seg in self.segments:
            norm = [normalize_fa(w["w"]) for w in seg["words"]]
            # a word may hold a half-space (می‌شود) that normalises to two words
            flat, owner = [], []
            for k, n in enumerate(norm):
                for piece in n.split():
                    flat.append(piece)
                    owner.append(k)
            n = len(want)
            k = 0
            while k + n <= len(flat):
                if flat[k:k + n] == want and (k == 0 or owner[k - 1] != owner[k]) \
                        and (k + n == len(flat) or owner[k + n] != owner[k + n - 1]):
                    out.append((seg["id"], owner[k], owner[k + n - 1] + 1))
                    k += n
                else:
                    k += 1
        return out

    def replace_all(self, find, text):
        """Replace every occurrence of `find` with `text`; -> (changed line ids, how many)."""
        text = " ".join((text or "").split())[:500]
        hits = self.find_words(find)
        if not hits or not text:
            return [], 0
        self._remember({s for s, _, _ in hits}, f"replacing «{_iso(find)}» everywhere")
        for seg_id, i, j in reversed(hits):     # from the end: earlier positions stay valid
            seg = self.segments[seg_id]
            ws = seg["words"]
            seg["words"] = ws[:i] + self._edited_words(text, ws[i]["t"], self.end_of(seg, j), ws[i:j]) + ws[j:]
            seg["touched"] = True
        changed = sorted({s for s, _, _ in hits})
        for k in changed:
            self.segments[k]["rev"] += 1
            self._mark(self.segments[k])
        self.stats["corrected"] += len(hits)
        if self.on_change:
            self.on_change()
        return changed, len(hits)

    def apply_auto(self, seg, i, j, text, who, conf):
        """
        Settle an unsure part with the other models' words: replace it (state "auto"), or,
        if they heard the same, confirm it (state "ok"). Each word notes who decided ("by")
        and, when replaced, what was there before ("was"), so the page can show and undo it.
        """
        ws = seg["words"]
        heard = " ".join(w["w"] for w in ws[i:j])
        by = f"{who} {round(conf * 100)}%"
        if " ".join(text.split()) == heard or _same_words(text, heard):
            new = [dict(w, s="ok", by=by) for w in ws[i:j]]
            self.stats["confirmed"] += 1
        else:
            t0, t1 = ws[i]["t"], self.end_of(seg, j)
            parts = text.split()
            step = (t1 - t0) / len(parts)
            new = [{"w": p, "t": round(t0 + k * step, 3), "c": round(conf, 3), "s": "auto",
                    "by": by, "was": heard} for k, p in enumerate(parts)]
            self.stats["auto"] += 1
        seg["words"] = ws[:i] + new + ws[j:]
        seg["touched"] = True
        seg["rev"] += 1
        self._mark(seg)
        if self.on_change:
            self.on_change()

    # -- saving / restoring edits ----------------------------------------------
    def export_edits(self):
        """The lines the user changed, to be saved and restored by a later run."""
        return {"format": 1, "stats": self.stats, "marks": self.marks, "place": round(self.place, 2),
                "edited": {str(s["id"]): [[w["w"], w["t"], w["c"], w["s"]]
                                          + ([w.get("by", ""), w.get("was", "")]
                                             if "by" in w or "was" in w else [])
                                          for w in s["words"]]
                           for s in self.segments if s["touched"]}}

    def restore_edits(self, state) -> int:
        """Put back saved edits (for lines that exist) and marks; returns how many lines."""
        n = 0
        try:
            self.place = float((state or {}).get("place", 0) or 0)
        except (TypeError, ValueError):
            self.place = 0.0
        self.marks = [{"a": float(m["a"]), "b": float(m["b"])}   # (older saves also had a kind: ignored)
                      for m in (state or {}).get("marks", [])
                      if isinstance(m, dict) and {"a", "b"} <= set(m)]
        for k, words in (state or {}).get("edited", {}).items():
            try:
                seg = self.segments[int(k)]
                restored = []
                for item in words:   # [word, time, conf, state] (+ [by, was] for changed words)
                    w, t, c, st = item[:4]
                    d = {"w": w, "t": float(t), "c": float(c), "s": st}
                    if len(item) >= 6:
                        if item[4]:
                            d["by"] = item[4]
                        if item[5]:
                            d["was"] = item[5]
                    restored.append(d)
                seg["words"] = restored
            except (ValueError, IndexError, TypeError):
                continue
            seg["touched"] = True
            seg["rev"] += 1
            self._mark(seg)
            n += 1
        for k, v in (state or {}).get("stats", {}).items():
            if k != "unsure" and k in self.stats and isinstance(v, int):
                self.stats[k] = v
        return n

    def reconfigure(self, threshold, fixer, kw_path):
        """New settings while open: lines you haven't edited are checked again."""
        self.threshold = threshold
        self.fixer = fixer
        self.kw_path = kw_path
        for seg in self.segments:
            if seg["touched"]:
                self._mark(seg)
            else:
                self._refix(seg)
            seg["rev"] += 1
        self.stats["unsure"] = max(self.stats["unsure"], self.left())

    def _append_keyword(self, line):
        header = "# ---------- added while reviewing ----------"
        text = self.kw_path.read_text(encoding="utf-8-sig") if self.kw_path.exists() else ""
        add = ""
        if text and not text.endswith("\n"):
            add += "\n"
        if header not in text:
            add += f"\n{header}\n"
        add += line + "\n"
        with open(self.kw_path, "a", encoding="utf-8") as f:
            f.write(add)
        self.fixer = self.fixer_factory(self.kw_path)

    def _refix_untouched(self):
        changed = []
        for seg in self.segments:
            if seg["touched"]:
                continue
            before = [w["w"] for w in seg["words"]]
            self._refix(seg)
            if [w["w"] for w in seg["words"]] != before:
                seg["rev"] += 1
                changed.append(seg["id"])
        return changed

    # -- output --------------------------------------------------------------
    def seg_json(self, seg):
        return {"id": seg["id"], "rev": seg["rev"],
                "start": round(seg["start"], 3), "end": round(seg["end"], 3),
                "words": [{"w": w["w"], "t": round(w["t"], 3), "c": round(w["c"], 3), "s": w["s"],
                           **({"u": w["u"]} if "u" in w else {}),
                           **{k: w[k] for k in ("by", "was", "p") if k in w}} for w in seg["words"]]}

    def counts(self):
        return {"left": self.left(), "total": self.stats["unsure"], "marks": self.marks,
                "auto": self.stats["auto"], "confirmed": self.stats["confirmed"],
                "transcribing": self.transcribing, "decoded": round(self.decoded_until, 1),
                "undo": self.undo_label()}

    def to_json(self, first=0, **extra):
        return {"file": self.name, "duration": self.duration, "kw_file": self.kw_path.name,
                "threshold": self.threshold, "place": round(self.place, 2),
                "segments": [self.seg_json(s) for s in self.segments[first:]],
                **self.counts(), **extra}

    def decoded(self):
        """The transcript as it stands: [(start, duration, [(word + punctuation, time, conf)])]."""
        out = []
        for seg in self.segments:
            words = [(w["w"] + w.get("p", ""), w["t"], w["c"]) for w in seg["words"]]
            if words:
                out.append((seg["start"], seg["end"] - seg["start"], words))
        return out

    def finish(self, rest: str):
        """-> ([(start, duration, [(word, time, conf)])], keyword fixes). rest: keep|blank."""
        fixes = sum(seg["fixes"] for seg in self.segments)
        if rest == "blank":
            for seg in self.segments:
                for i, j in reversed(self.spans(seg)):
                    self.apply_edit(seg["id"], i, j, "blank")
        return self.decoded(), fixes

    def question(self, seg, i, j, n):
        ws = seg["words"]
        heard = " ".join(w["w"] for w in ws[i:j])
        return {
            "file": self.name, "n": n, "time": _fmt_time(ws[i]["t"]),
            "conf": min(w["c"] for w in ws[i:j]),
            "before": " ".join(w["w"] for w in ws[max(0, i - CONTEXT_WORDS):i]),
            "heard": heard,
            "after": " ".join(w["w"] for w in ws[j:j + CONTEXT_WORDS]),
            "rule_ok": rule_allowed(heard), "kw_file": self.kw_path.name,
        }


# --------------------------------------------------------------------------- terminal UI
class Player:
    def __init__(self):
        self._dir = Path(tempfile.gettempdir())
        self._n = 0

    def play(self, wav: bytes) -> None:
        # Alternate between two files so a clip still playing is never overwritten.
        self._n ^= 1
        path = self._dir / f"goosh_o_ghalam_clip{self._n}.wav"
        path.write_bytes(wav)
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            print(f"  (clip saved to {path})")

    def stop(self) -> None:
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)


class TerminalUI:
    """Ask about each unsure part in the console (Persian may look broken in the Windows console)."""

    def __init__(self, player=None, ask=input, say=print):
        self.player = player if player is not None else Player()
        self.input = ask
        self.say = say

    def _read(self):
        try:
            return self.input("  > ").strip()
        except EOFError:
            return "q"

    def ask(self, q, wav):
        """-> (action, text, add); action may also be "stop"."""
        self.say("")
        self.say(f"  -- {q['time']}  unsure #{q['n']}  (confidence {q['conf']:.0%})")
        self.say(f"  {q['before']} [ ... ] {q['after']}")
        self.say(f"  heard:  {q['heard']}")
        self.say("  Enter = keep  |  type the right words  |  r = replay  b = blank  d = delete  q = stop asking")
        while True:
            self.player.play(wav)
            reply = self._read()
            if reply.lower() != "r":
                break
        self.player.stop()

        low = reply.lower()
        if reply == "" or reply == q["heard"]:
            return "keep", None, None
        if low in ("b", "d"):
            return {"b": "blank", "d": "delete"}[low], None, None
        if low == "q":
            return "stop", None, None
        if q["rule_ok"]:
            self.say(f"  Add to {q['kw_file']}?  k = keyword «{reply}»  |  "
                     f"r = rule «{q['heard']} => {reply}»  |  Enter = no")
        else:
            self.say(f"  Add to {q['kw_file']}?  k = keyword «{reply}»  |  Enter = no"
                     f"   (no rule: «{q['heard']}» is too short, it would be changed everywhere)")
        add = {"k": "keyword", "r": "rule"}.get(self._read().lower())
        return "replace", reply, add

    def notice(self, text):
        self.say("  " + text.replace("⁨", "").replace("⁩", ""))

    def run(self, session, blank_rest):
        """Ask about each unsure part in order. Returns what to do with the rest: keep|blank."""
        n = 0
        while True:
            nxt = session.first_unsure()
            if nxt is None:
                return "keep"
            seg, i, j = nxt
            n += 1
            action, text, add = self.ask(session.question(seg, i, j, n), session.clip_for(seg, i, j))
            if action == "stop":
                session.stopped = True
                rest = "blank" if blank_rest else "keep"
                self.notice(f"Stopped asking; the remaining unsure parts will be "
                            f"{'blanked' if rest == 'blank' else 'kept as heard'}.")
                return rest
            res = session.apply_edit(seg["id"], i, j, action, text, add)
            if res["notice"]:
                self.notice(res["notice"])

    def status(self, text, progress=None):
        pass

    def set_file(self, name):
        pass

    def close(self, summary):
        pass


# --------------------------------------------------------------------------- browser UI
class WebUI:
    """
    Serves review_page.html on 127.0.0.1 and opens it in the browser. The transcript is
    shown while the file is still being transcribed; new lines are fetched as they come.
      GET  /state      status, file id, number of lines so far
      GET  /doc        ?from=<n>: lines n.. of the transcript (all by default) and counts
      GET  /clip.wav   ?a=<sec>&b=<sec> audio of the file under review
      POST /edit       {seg, i, j, action, text, add} -> changed lines
      POST /finish     {rest: keep|blank} -> write the subtitles (once transcribed)
      GET  /fs         ?path=<folder>: its subfolders and recordings (with how far each got)
      POST /open       {path}: switch to that recording (the current one stays saved)
      POST /quit       close the program
    """

    def __init__(self, open_browser=True, alts=None, media_exts=(), work_root=None, model=("", "", ""),
                 auto_replace=None, model_dir=None, threads=4):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent / "models"
        self.threads = threads
        self.punctuator = None             # punctuation.Punctuator, loaded on first use
        self.speakers = None               # {"turns": [[a, b, speaker]], "names": {id: name}} of the open file
        self.itn_for = None                # -> spoken-numbers-to-digits function or None (set by the program)
        self.batch = None                  # transcribe.BatchWorker: the background queue (set by the program)
        self.last_seen = None              # when a page last asked for /state
        self.bye_at = None                 # when a page last said it was closing
        self._place_saved = 0.0
        self.alts = alts   # alternatives.AltEngine: other models' guesses, or None
        if alts is not None:
            alts.wanted = self._alt_wanted
            alts.on_result = self._on_alt_result
        # settle unsure parts with the other models when they're this sure (None: never)
        self.auto_replace = auto_replace if alts is not None else None
        self.work = None
        self.use_alts = alts is not None   # this recording uses the other models
        self.settings_for = None           # path -> (settings, saved?)      (set by the program)
        self.apply_settings = None         # settings -> (fixer, list for additions)
        self.keyword_lists = None          # -> {list name: entries}
        self.picked_settings = None        # settings chosen with the last /open or /redo
        self.open_key = None   # the part open in the editor bar: never changed automatically
        self.changes = []      # (sequence number, line id) of lines changed automatically
        self.media_exts = {e.lower() for e in media_exts}
        self.work_root = work_root
        self.token = secrets.token_urlsafe(16)
        self.lock = threading.Lock()
        self.state = {"status": "Loading the model…", "file": "", "path": "", "file_id": 0,
                      "lines": 0, "reviewing": False, "transcribing": False, "decoded": 0,
                      "duration": 0, "done": False, "browsing": False, "folder": "", "progress": None,
                      "model": model[0], "model_about": model[1], "model_url": model[2], "change_seq": 0,
                      "auto_replace": auto_replace if alts is not None else None,
                      "job": None,        # {"kind", "text", "progress"} while punctuation etc. run
                      "job_done": None,   # {"seq", "text"}: the last one's outcome, for a message
                      "doc_rev": 0,       # goes up when the whole transcript should be fetched again
                      "queue": None}      # the background queue's progress (see BatchWorker)
        self.session = None
        self.blank_default = False
        self.finished = threading.Event()
        self.rest = "keep"
        self.pending = None          # ("open", path) or ("quit", None) asked for on the page
        self.wake = threading.Event()
        # A fixed port when free, so the page keeps its settings (e.g. playback speed)
        # from one run to the next; any free port otherwise.
        handler = _make_handler(self)
        self.server = None
        for port in list(range(REVIEW_PORT, REVIEW_PORT + 5)) + [0]:
            try:
                self.server = _ExclusiveServer(("127.0.0.1", port), handler)
                break
            except OSError:
                continue
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/?t={self.token}"
        print(f"Goosh-o-Ghalam review page: {self.url}")
        print("  (the transcript appears there while it is transcribed; keep this window open."
              " Closing the page closes the program too, a little later)")
        threading.Thread(target=self._watchdog, daemon=True).start()
        if open_browser:
            webbrowser.open(self.url)

    def _watchdog(self):
        """Close the program once no review page is open any more (everything is saved anyway)."""
        while True:
            time.sleep(5)
            with self.lock:
                if self.state["done"]:
                    return
                now = time.time()
                q = self.state.get("queue") or {}
                if self.last_seen is None or self.pending is not None or q.get("current") or q.get("waiting"):
                    continue     # (a queue left running overnight keeps the program open)
                said_bye = self.bye_at is not None and self.bye_at > self.last_seen and now - self.bye_at > PAGE_BYE
                if not (said_bye or now - self.last_seen > PAGE_GONE):
                    continue
                print("\nThe review page was closed: closing the program (everything is saved).")
                self.pending = ("quit", None)
                self.wake.set()
            if self.batch:
                self.batch.stop()

    # -- jobs that run beside the review (punctuation, speakers) ---------------------
    def _start_job(self, kind, text, fn):
        """Run fn() in the background (under self.lock); its return value is shown when done."""
        if self.state["job"]:
            raise ValueError(f"wait until this is done: {self.state['job']['text']}")
        self.state["job"] = {"kind": kind, "text": text, "progress": None}

        def run():
            try:
                msg = fn()
            except Exception as e:   # shown on the page
                msg = f"{kind.capitalize()} failed: {e}"
                print(f"  {msg}")
            with self.lock:
                self.state["job"] = None
                seq = (self.state["job_done"] or {}).get("seq", 0) + 1
                self.state["job_done"] = {"seq": seq, "text": msg}
        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def _job_progress(self, text, frac=None):
        with self.lock:
            if self.state["job"]:
                self.state["job"].update(text=text, progress=frac)

    def punctuate(self, auto=False):
        """Add punctuation to the open transcript (under self.lock)."""
        s, work = self.session, self.work
        if s is None:
            raise ValueError("nothing open")

        def fn():
            import punctuation
            if punctuation.missing(self.model_dir):
                punctuation.download(self.model_dir, self._job_progress)
            self._job_progress("Adding punctuation: loading the model…")
            if self.punctuator is None:
                self.punctuator = punctuation.Punctuator(self.model_dir, self.threads)
            with self.lock:
                lines = len(s.segments)
                items = [(seg["id"], w["t"], w["w"]) for seg in s.segments
                         for w in seg["words"] if w["s"] != "blank"]
            marks = self.punctuator.marks(
                [x[2] for x in items], lambda f: self._job_progress(f"Adding punctuation… {f:.0%}", f))
            out = {}
            for (sid, t, _), m in zip(items, marks):
                if m:
                    out.setdefault(sid, {})[f"{t:.3f}"] = m
            with self.lock:
                s.set_punct(out, lines)      # saved by the session's on_punct
                if self.session is s:
                    self.state["doc_rev"] += 1
            n = sum(len(v) for v in out.values())
            return f"Punctuation added ({n} marks). Click a mark's word to change it."
        return self._start_job("punctuation", "Adding punctuation…", fn)

    def find_speakers(self, data):
        """Work out who speaks when in the open recording (under self.lock)."""
        s, work = self.session, self.work
        if s is None or work is None:
            raise ValueError("nothing open")
        n = data.get("n")
        if n in ("", None, 0):
            n = None
        elif not (isinstance(n, int) and 1 <= n <= 50):
            raise ValueError("number of speakers: 1 to 50, or empty to find out")

        def fn():
            import speakers
            if speakers.missing(self.model_dir):
                speakers.download(self.model_dir, self._job_progress)
            self._job_progress("Finding who speaks when…", 0.0)
            turns = speakers.find_speakers(
                s.audio, self.model_dir, n, self.threads,
                lambda f: self._job_progress(f"Finding who speaks when… {f:.0%}", f))
            count = len({k for *_, k in turns})
            data = {"turns": [[round(a, 2), round(b, 2), k] for a, b, k in turns], "names": {},
                    "asked": n}
            work.save_json("speakers.json", data)
            with self.lock:
                if self.work is work:
                    self.speakers = data
                    self.state["doc_rev"] += 1
            return (f"Found {count} speaker{'s' if count != 1 else ''}. "
                    "Click a speaker's name in the text to rename them.")
        return self._start_job("speakers", "Finding who speaks when…", fn)

    def name_speaker(self, data):
        if not self.speakers or self.work is None:
            raise ValueError("no speakers found yet")
        k, name = data.get("id"), " ".join(str(data.get("name") or "").split())[:40]
        if not isinstance(k, int) or k < 0:
            raise ValueError("bad speaker")
        names = self.speakers.setdefault("names", {})
        if name:
            names[str(k)] = name
        else:
            names.pop(str(k), None)
        self.work.save_json("speakers.json", self.speakers)
        return {"ok": True, "speakers": self.speakers}

    def export(self):
        """Write the transcript as it stands as .md and .docx (under self.lock)."""
        s, work = self.session, self.work
        if s is None or work is None:
            raise ValueError("nothing open")
        import export
        sp = self.speakers or {}
        itn = self.itn_for() if self.itn_for else None
        files = export.write_documents(work.srt.with_suffix(""), Path(s.name).stem, s.decoded(), s.marks,
                                       [tuple(t) for t in sp.get("turns", [])], sp.get("names"), itn)
        return {"files": [f.name for f in files], "folder": str(work.dir), "first": str(files[-1])}

    def reveal(self, data):
        """Show a file (or the recording's folder) in Explorer."""
        target = Path(str(data.get("path") or "")) if data.get("path") else (self.work.dir if self.work else None)
        if target is None or not target.exists():
            raise ValueError("nothing to show")
        if os.name == "nt":
            import subprocess
            if target.is_file():
                subprocess.Popen(["explorer", "/select,", str(target)])
            else:
                os.startfile(str(target))
        return {"ok": True}

    def set_place(self, data):
        s = self.session
        if s is None:
            return {"ok": False}
        try:
            s.place = max(0.0, min(float(data.get("t", 0)), s.duration))
        except (TypeError, ValueError):
            raise ValueError("bad time")
        if s.on_change and (data.get("flush") or time.time() - self._place_saved > 10):
            self._place_saved = time.time()
            s.on_change()
        return {"ok": True}

    def undo(self):
        s = self.session
        if s is None:
            raise ValueError("nothing under review")
        changed, what = s.undo()
        reload = len(changed) > 50
        return {"notice": f"Undid {what}", "reload": reload, **s.counts(),
                "segments": [] if reload else [s.seg_json(s.segments[k]) for k in changed]}

    def replace_all(self, data):
        s = self.session
        if s is None:
            raise ValueError("nothing under review")
        find, text = str(data.get("find") or ""), str(data.get("text") or "")
        changed, n = s.replace_all(find, text)
        return {"n": n, **s.counts(), "segments": [s.seg_json(s.segments[k]) for k in changed]}

    def search(self, query, scope):
        """GET /search: lines with `query`, in the open transcript (this) or every saved one (all)."""
        from workdir import normalize_fa, search
        if scope == "all":
            return {"scope": "all", **search(self.work_root, query)}
        s = self.session
        q = normalize_fa(query)
        hits = []
        if s is None or len(q) < 2:
            return {"scope": "this", "hits": hits}
        with self.lock:
            for seg in s.segments:
                norm = [normalize_fa(w["w"]) for w in seg["words"]]
                text, starts = "", []
                for n in norm:
                    starts.append(len(text) + (1 if text else 0))
                    text = f"{text} {n}" if text else n
                pos = text.find(q)
                while pos >= 0 and len(hits) < 500:
                    i = max(k for k, st in enumerate(starts) if st <= pos)
                    j = max(k for k, st in enumerate(starts) if st < pos + len(q)) + 1
                    hits.append({"seg": seg["id"], "i": i, "j": j, "t": seg["words"][i]["t"],
                                 "text": " ".join(w["w"] for w in seg["words"])})
                    pos = text.find(q, pos + len(q))
            whole = len(s.find_words(query))   # as whole words: what "Replace all" would change
        return {"scope": "this", "hits": hits, "whole": whole}

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def configure(self, fs):
        """Settings of the recording about to open (or open): second opinions, auto-replace."""
        self.use_alts = self.alts is not None and bool(fs.get("alternatives"))
        self.auto_replace = fs.get("auto_replace") if self.use_alts else None
        self.state["settings"] = dict(fs)

    def file_settings(self, path):
        """GET /file_settings: what the "Before starting" dialog shows (outside the lock)."""
        fs, saved = self.settings_for(path)
        lists = self.keyword_lists()
        return {"settings": fs, "saved": saved, "has_alts": self.alts is not None,
                "lists": [{"name": n, "entries": c} for n, c in lists.items()]}

    def live_settings(self, data):
        """POST /settings: change the open recording's settings (under the lock)."""
        if self.session is None or self.apply_settings is None:
            raise ValueError("nothing open")
        fixer, kw_path = self.apply_settings(data.get("settings"))
        fs = self.state.get("settings", {})
        fs.update(self.settings_for(self.state["path"])[0])
        self.configure(fs)
        s = self.session
        s.reconfigure(fs["confidence"], fixer, kw_path)
        if self.alts and self.use_alts:
            fid = self.state["file_id"]
            for seg in s.segments:
                self._auto_check(seg)
                for i, j in s.spans(seg):
                    key, samples, *_ = s.alt_request(seg, i, j)
                    self.alts.request((fid,) + key, samples)
        return {"ok": True, "settings": fs}

    def status(self, text, progress=None):
        """What the page shows while no transcript is up; progress 0..1 adds a bar."""
        self._set(status=text, progress=progress)

    def set_file(self, name):
        self._set(file=name)

    # -- called from the transcriber (main thread) ----------------------------
    def attach(self, session, blank_rest, work=None):
        """Show a file (with any lines already transcribed) on the page."""
        with self.lock:
            self.session = session
            self.blank_default = blank_rest
            self.rest = "blank" if blank_rest else "keep"
            self.finished.clear()
            fid = self.state["file_id"] + 1
            self.state.update(file=session.name, file_id=fid, lines=len(session.segments),
                              reviewing=True, transcribing=True, browsing=False,
                              decoded=round(session.decoded_until, 1),
                              duration=round(session.duration, 1), status="Transcribing…", progress=None)
            if work is not None:
                self.state.update(path=str(work.media.resolve()), folder=str(work.media.resolve().parent))
            self.work = work
            self.open_key = None
            self.changes = []
            self.speakers = None
            if work is not None:
                session.restore_punct(work.load_json("punctuation.json"))
                session.on_punct = lambda: work.save_json("punctuation.json", session.export_punct())
                sp = work.load_json("speakers.json")
                self.speakers = sp if isinstance(sp, dict) and isinstance(sp.get("turns"), list) else None
            if self.alts and self.use_alts:
                if work is not None:  # guesses saved by an earlier run
                    self.alts.preload({(fid,) + k: v for k, v in work.load_alternatives().items()})
                for seg in session.segments:
                    self._auto_check(seg)
                    for i, j in session.spans(seg):
                        key, samples, *_ = session.alt_request(seg, i, j)
                        self.alts.request((fid,) + key, samples)

    def add_segment(self, session, start, dur, words):
        with self.lock:
            seg = session.add_segment(start, dur, words)
            self.state.update(lines=len(session.segments), decoded=round(session.decoded_until, 1))
            if self.alts and self.use_alts:  # get the other models' guesses ready before they're asked for
                self._auto_check(seg)   # (results saved by an earlier run may settle it now)
                for i, j in session.spans(seg):
                    key, samples, *_ = session.alt_request(seg, i, j)
                    self.alts.request((self.state["file_id"],) + key, samples)

    # -- settling unsure parts with the other models ------------------------------
    def _on_alt_result(self, key, name, result):
        """A model finished a clip (called from its thread): save it, then try to settle."""
        with self.lock:
            if self.session is None or key[0] != self.state["file_id"]:
                return
            work = self.work
            a, b = key[1], key[2]
            for seg in self.session.segments:
                if seg["start"] <= b and seg["end"] >= a:
                    self._auto_check(seg)
        if work is not None:
            work.save_alternative(a, b, name, result)

    def _auto_check(self, seg):
        """Settle the unsure parts of a line whose clips the other models agree on (under the lock)."""
        if self.auto_replace is None or self.alts is None or self.session is None:
            return
        from alternatives import auto_decision
        s = self.session
        for i, j in reversed(s.spans(seg)):     # from the end: earlier indices stay valid
            key, _, before, span, after = s.alt_request(seg, i, j)
            key = (self.state["file_id"],) + key
            if key == self.open_key:
                continue
            got = self.alts.get(key)
            if not got:
                continue
            main_conf = min(w["c"] for w in seg["words"][i:j])
            decision = auto_decision(before, span, after, got, self.auto_replace, main_conf)
            if decision:
                s.apply_auto(seg, i, j, *decision)
                self.state["change_seq"] += 1
                self.changes.append((self.state["change_seq"], seg["id"]))

    def changes_since(self, seq):
        """Lines changed automatically since `seq` (under the lock)."""
        s = self.session
        ids = sorted({sid for n, sid in self.changes if n > seq})
        return {"seq": self.state["change_seq"],
                "segments": [s.seg_json(s.segments[k]) for k in ids] if s else [],
                **(s.counts() if s else {})}

    def _alt_wanted(self, key):
        """Is a prepared clip still worth computing? (called by the AltEngine threads)"""
        with self.lock:
            s = self.session
            if s is None or key[0] != self.state["file_id"]:
                return False
            ab = tuple(key[1:])
            return any(s.alt_key(seg, i, j) == ab for seg in s.segments for i, j in s.spans(seg))

    def alternatives(self, data):
        """Other models' guesses for words[i:j] of a segment (under self.lock)."""
        s = self.session
        if s is None or self.alts is None or not self.use_alts:
            return {"models": []}
        seg_id, i, j = data.get("seg"), data.get("i"), data.get("j")
        if not (0 <= seg_id < len(s.segments) and 0 <= i < j <= len(s.segments[seg_id]["words"])):
            raise ValueError("bad word range")
        from alternatives import NOW, SOON, align
        key, samples, before, span, after = s.alt_request(s.segments[seg_id], i, j)
        key = (self.state["file_id"],) + key
        self.open_key = key
        self.alts.request(key, samples, NOW)
        # the parts the user will most likely open next go ahead of the rest of the file
        for seg, a, b in s.unsure_after(seg_id, i, 3):
            k2, samples2, *_ = s.alt_request(seg, a, b)
            self.alts.request((self.state["file_id"],) + k2, samples2, SOON)
        out = []
        for r in self.alts.results(key):
            if r["words"] is not None:
                r["match"] = align(before, span, after, r["words"])
            out.append(r)
        return {"models": out}

    def progress(self, sec):
        self._set(decoded=round(sec, 1))

    def run(self, session, blank_rest):
        """Transcription is done: wait for Finish (it may already have been pressed)."""
        with self.lock:
            session.transcribing = False
            self.state.update(transcribing=False, decoded=round(session.duration, 1),
                              status="Transcribed", progress=None)
            # punctuation, if not done yet for every line (it takes about 20 s an hour)
            if not self.state["job"] and session.punct_lines < len(session.segments) and session.segments:
                try:
                    self.punctuate(auto=True)
                except ValueError:
                    pass
        if not self.finished.is_set():
            print("  Transcribed. Finish the review in the browser (press Finish there).")
        while True:
            try:
                if self.finished.wait(0.5):
                    break
            except KeyboardInterrupt:
                print("\n  Ctrl+C: finishing this file "
                      f"(unreviewed unsure parts {'blanked' if blank_rest else 'kept'}).")
                break
            self.check_switch()
        with self.lock:
            self.session = None
            self.state.update(reviewing=False, status="Saving subtitles…", progress=None)
            return self.rest

    # -- choosing files on the page ------------------------------------------------
    def check_switch(self):
        """Raise SwitchFile if another file (or quitting) was asked for on the page."""
        with self.lock:
            if self.pending is None:
                return
            self.session = None
            self.state.update(reviewing=False, transcribing=False, status="Opening…", progress=None)
        raise SwitchFile()

    def take_pending(self):
        with self.lock:
            p, self.pending = self.pending, None
            self.wake.clear()
            return p or ("browse", None)

    def wait_for_file(self, status="Choose a recording"):
        """Show the file browser until a recording is picked: ("open", path) or ("quit", None)."""
        with self.lock:
            self.session = None
            self.state.update(reviewing=False, transcribing=False, browsing=True, status=status,
                              progress=None)
        while True:
            try:
                if self.wake.wait(0.5) and self.pending:
                    break
            except KeyboardInterrupt:
                with self.lock:
                    self.pending = ("quit", None)
                break
        with self.lock:
            self.state.update(browsing=False, status="Opening…", progress=None)
        return self.take_pending()

    def set_folder(self, folder):
        self._set(folder=str(folder))

    def list_folder(self, path):
        """A folder's subfolders and recordings (called outside the lock: it reads the disk)."""
        import string
        from workdir import adopt_old_folder, status
        p = Path(path or self.state["folder"] or Path.home()).expanduser()
        try:
            p = p.resolve()
            for e in list(p.iterdir()):   # progress folders left here by an earlier version
                if e.suffix.lower() in self.media_exts and (p / f"{e.stem}_transcript").is_dir():
                    adopt_old_folder(e, self.work_root)
            entries = sorted(p.iterdir(), key=lambda e: e.name.lower())
        except OSError as e:
            raise ValueError(f"can't open {p}: {e.strerror or e}")
        dirs, files = [], []
        for e in entries:
            try:
                if e.is_dir():
                    if not e.name.startswith((".", "$")):
                        dirs.append(e.name)
                elif e.suffix.lower() in self.media_exts:
                    files.append({"name": e.name, "path": str(e), "size": e.stat().st_size,
                                  "queue": self.batch.queued(e) if self.batch else "",
                                  **status(e, self.work_root)})
            except OSError:
                continue
        drives = ([f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
                  if os.name == "nt" else ["/"])
        self.set_folder(p)
        return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
                "dirs": dirs, "files": files, "drives": drives, "current": self.state["path"]}

    def queue_add(self, data):
        """POST /queue {folder} or {paths}: transcribe those recordings in the background (outside the lock)."""
        if self.batch is None:
            raise ValueError("the background queue isn't available")
        if isinstance(data.get("paths"), list):
            files = [Path(str(p)) for p in data["paths"]]
        else:
            folder = Path(str(data.get("folder") or ""))
            if not folder.is_dir():
                raise ValueError("not a folder")
            files = sorted((e for e in folder.iterdir() if e.suffix.lower() in self.media_exts),
                           key=lambda e: e.name.lower())
        open_now = self.state["path"]
        files = [f for f in files if f.is_file() and f.suffix.lower() in self.media_exts
                 and str(f.resolve()) != open_now]
        return {"added": self.batch.add(files)}

    # (these two are called outside self.lock: the queue reports back through self._set)
    def queue_stop(self):
        if self.batch is None:
            raise ValueError("the background queue isn't available")
        self.batch.stop()
        return {"ok": True}

    def request(self, data, action):
        """POST /open, /redo (start a recording over from scratch) or /quit (under self.lock)."""
        if action == "quit":
            self.pending = ("quit", None)
        else:
            path = Path(str(data.get("path", "")))
            if not path.is_file() or path.suffix.lower() not in self.media_exts:
                raise ValueError("not a recording")
            if action == "open" and str(path) == self.state["path"] and self.session is not None:
                return {"ok": True, "same": True}
            self.pending = ("redo" if action == "redo" else "open", str(path))
            chosen = data.get("settings")
            self.picked_settings = chosen if isinstance(chosen, dict) else None
        self.wake.set()
        return {"ok": True}

    # -- called from the server threads, under self.lock -----------------------
    def doc(self, first=0):
        s = self.session
        if s is None:
            return {"file_id": self.state["file_id"], "segments": None}
        return s.to_json(first=first, file_id=self.state["file_id"],
                         change_seq=self.state["change_seq"], auto_replace=self.auto_replace,
                         blank_default=self.blank_default, alts=self.alts is not None and self.use_alts,
                         speakers=self.speakers if first == 0 else None)

    def edit(self, data):
        s = self.session
        if s is None:
            raise ValueError("nothing under review")
        res = s.apply_edit(data.get("seg"), data.get("i"), data.get("j"), data.get("action"),
                           data.get("text") if isinstance(data.get("text"), str) else None,
                           data.get("add") if data.get("add") in ("keyword", "rule") else None)
        return {"notice": res["notice"], **s.counts(),
                "segments": [s.seg_json(s.segments[k]) for k in res["changed"]]}

    def mark(self, data):
        """POST /mark {a, b, on}: mark [a, b) as off-topic (on: false unmarks it)."""
        s = self.session
        if s is None:
            raise ValueError("nothing under review")
        try:
            a, b = float(data.get("a")), float(data.get("b"))
        except (TypeError, ValueError):
            raise ValueError("bad time range")
        s.set_mark(a, b, bool(data.get("on")))
        return s.counts()

    def finish(self, data):
        if self.session is None:
            raise ValueError("nothing under review")
        self.rest = "blank" if data.get("rest") == "blank" else "keep"
        self.finished.set()
        if self.session.transcribing:
            self.state["status"] = "Review finished. The subtitles are saved once transcription completes…"
        return {"ok": True}

    def close(self, summary):
        self._set(done=True, reviewing=False, status=summary, progress=None)
        time.sleep(1.5)  # let the page see it before the server goes away
        self.server.shutdown()


class SwitchFile(Exception):
    """Another recording (or quitting) was chosen on the page; the current one stays saved."""


class _ExclusiveServer(ThreadingHTTPServer):
    """Fails if the port is taken (Windows would otherwise let two servers share it)."""
    allow_reuse_address = False

    def server_bind(self):
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _make_handler(ui):
    page = PAGE.read_text(encoding="utf-8").encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _send_audio(self, pcm: bytes):
            """The whole file as a WAV, in byte ranges so the browser can seek."""
            header = _wav_header(len(pcm))
            total = len(header) + len(pcm)
            start, end = 0, total - 1
            rng = self.headers.get("Range", "")
            partial = rng.startswith("bytes=")
            if partial:
                try:
                    a, b = rng[6:].split(",")[0].strip().split("-")
                    if a:
                        start = int(a)
                        end = int(b) if b else total - 1
                    else:  # last N bytes
                        start = max(0, total - int(b))
                except ValueError:
                    partial = False
                    start, end = 0, total - 1
            if start >= total or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.end_headers()
                return
            end = min(end, total - 1, start + AUDIO_CHUNK - 1)  # the browser asks for more
            body = (header + pcm[:max(0, end + 1 - len(header))])[start:] if start < len(header) \
                else pcm[start - len(header):end + 1 - len(header)]
            self.send_response(206 if partial or end < total - 1 else 200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            if partial or end < total - 1:
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the browser seeks by dropping a request half-way

        def _authorised(self, url):
            token = parse_qs(url.query).get("t", [""])[0] or self.headers.get("X-Token", "")
            return secrets.compare_digest(token, ui.token)

        def do_GET(self):
            url = urlparse(self.path)
            if not self._authorised(url):
                return self._send(403, b"Open the link printed in the terminal.", "text/plain")
            if url.path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if url.path == "/audio.wav":
                with ui.lock:
                    session = ui.session
                if session is None:
                    return self._send(404, b"nothing under review", "text/plain")
                return self._send_audio(session.pcm())
            if url.path == "/file_settings":
                try:
                    return self._json(ui.file_settings(parse_qs(url.query).get("path", [""])[0]))
                except (ValueError, OSError) as e:
                    return self._json({"error": str(e)}, 400)
            if url.path == "/fs":
                try:
                    return self._json(ui.list_folder(parse_qs(url.query).get("path", [""])[0]))
                except ValueError as e:
                    return self._json({"error": str(e)}, 400)
            if url.path == "/search":      # (reads saved transcripts from disk: outside the lock)
                qs = parse_qs(url.query)
                return self._json(ui.search(qs.get("q", [""])[0], qs.get("scope", ["this"])[0]))
            with ui.lock:
                if url.path == "/state":   # (+ how the other models' setup is going)
                    ui.last_seen = time.time()
                    return self._json({**ui.state,
                                       "alts_status": dict(ui.alts.status) if ui.alts else {}})
                if url.path == "/doc":
                    try:
                        first = max(0, int(parse_qs(url.query).get("from", ["0"])[0]))
                    except ValueError:
                        first = 0
                    return self._json(ui.doc(first))
                if url.path == "/changes":
                    try:
                        since = int(parse_qs(url.query).get("since", ["0"])[0])
                    except ValueError:
                        since = 0
                    return self._json(ui.changes_since(since))
                if url.path == "/alts":
                    qs = parse_qs(url.query)
                    try:
                        req = {k: int(qs.get(k, ["-1"])[0]) for k in ("seg", "i", "j")}
                        return self._json(ui.alternatives(req))
                    except ValueError as e:
                        return self._json({"error": str(e) or "bad request"}, 400)
                if url.path == "/clip.wav" and ui.session is not None:
                    qs = parse_qs(url.query)
                    try:
                        a = float(qs.get("a", ["0"])[0])
                        b = float(qs.get("b", ["0"])[0])
                    except ValueError:
                        return self._send(400, b"bad range", "text/plain")
                    return self._send(200, ui.session.clip(a, b), "audio/wav")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            url = urlparse(self.path)
            routes = {
                "/edit": ui.edit, "/mark": ui.mark, "/settings": ui.live_settings, "/finish": ui.finish,
                "/undo": lambda d: ui.undo(), "/replace_all": ui.replace_all,
                "/speakers": ui.find_speakers,
                "/speaker_name": ui.name_speaker, "/export": lambda d: ui.export(), "/reveal": ui.reveal,
                "/place": ui.set_place, "/queue": ui.queue_add, "/queue_stop": lambda d: ui.queue_stop(),
                "/open": lambda d: ui.request(d, "open"), "/redo": lambda d: ui.request(d, "redo"),
                "/quit": lambda d: ui.request(d, "quit"),
            }
            if not self._authorised(url) or (url.path not in routes and url.path != "/bye"):
                return self._send(403, b"forbidden", "text/plain")
            try:
                n = min(int(self.headers.get("Content-Length", 0)), 100_000)
                raw = self.rfile.read(n).decode("utf-8")
                if url.path == "/bye":    # the page is closing (sent with navigator.sendBeacon)
                    with ui.lock:
                        ui.bye_at = time.time()
                        try:
                            place = json.loads(raw or "{}").get("t")
                            if place is not None:
                                ui.set_place({"t": place, "flush": True})
                        except (ValueError, AttributeError):
                            pass
                    return self._json({"ok": True})
                data = json.loads(raw or "{}")
                if not isinstance(data, dict):
                    raise ValueError("bad request")
                if url.path in ("/queue", "/queue_stop"):    # (the queue takes the lock itself)
                    res = routes[url.path](data)
                else:
                    with ui.lock:
                        res = routes[url.path](data)
            except (ValueError, UnicodeDecodeError, OSError) as e:
                return self._json({"error": str(e) or "bad request"}, 400)
            return self._json(res)

    return Handler


# --------------------------------------------------------------------------- reviewer
class Reviewer:
    """
    For each file: build a ReviewSession, then
      mode "ask":   let the user review it (browser page or terminal)
      mode "blank": replace unsure parts with […] without asking
    Keyword-list additions carry over to the following files.
    """

    def __init__(self, mode: str, threshold: float, kw_path: Path, fixer, fixer_factory,
                 blank_on_stop: bool = False, ui=None):
        self.mode = mode
        self.threshold = threshold
        self.kw_path = kw_path
        self.fixer = fixer
        self.fixer_factory = fixer_factory   # builds a KeywordFixer from kw_path
        self.blank_on_stop = blank_on_stop
        self.ui = ui if ui is not None else (TerminalUI() if mode == "ask" else None)
        self.stopped = False   # terminal: q stops asking for the rest of the run
        self.saved = []        # one line per finished file, shown at the end
        self.stats = {}
        self.session = None
        self.live = False
        self.marks = []        # off-topic stretches of the file just finished
        self.work = None

    def set_file(self, name):
        if self.ui:
            self.ui.set_file(name)

    def status(self, text, progress=None):
        if self.ui:
            self.ui.status(text, progress)

    # -- one file: start_file, add_segment (while transcribing), finish_file -----
    def start_file(self, name, duration, audio, work=None, cached=()):
        """
        cached: lines an earlier run already transcribed; work: workdir.WorkDir, where the
        review edits (and the other models' guesses) are saved and restored from.
        """
        self.work = work
        session = self.session = ReviewSession(name, duration, audio, self.threshold, self.fixer,
                                               self.kw_path, self.fixer_factory, decoded=cached)
        if work is not None and self.mode == "ask":
            n = session.restore_edits(work.load_review())
            if n:
                print(f"  restored your edits on {n} line(s) from the last review")
            session.on_change = lambda: work.save_review(session.export_edits())
        self.live = self.mode == "ask" and not self.stopped and hasattr(self.ui, "attach")
        if self.live:
            self.ui.attach(session, self.blank_on_stop, work)
        else:
            self.set_file(name)
            self.status("Transcribing…")

    def add_segment(self, start, dur, words):
        if self.live:
            self.ui.add_segment(self.session, start, dur, words)
        else:
            self.session.add_segment(start, dur, words)

    def progress(self, sec):
        if self.live:
            self.ui.progress(sec)

    def check_switch(self):
        """Raises SwitchFile if another recording was opened on the page."""
        if self.live:
            self.ui.check_switch()

    def abandon_file(self):
        """Leave the current file as saved (its edits are already on disk)."""
        if self.work is not None and self.session is not None and self.session.on_change:
            self.session.on_change()
        self.session = None
        self.stats = {}
        self.marks = []

    def finish_file(self):
        """-> (reviewed segments [(start, dur, [(word, time, conf)])], keyword fixes)."""
        session = self.session
        if self.mode == "blank" or self.stopped:
            rest = "blank" if (self.mode == "blank" or self.blank_on_stop) else "keep"
        else:
            rest = self.ui.run(session, self.blank_on_stop)
            self.stopped = session.stopped
        self.fixer = session.fixer
        saving = session.on_change
        session.on_change = None             # save once below, not after every blank
        out, fixes = session.finish(rest)
        if saving:
            saving()
        self.stats = session.stats
        self.marks = list(session.marks)     # off-topic stretches, for the lesson-only files
        self.session = None
        return out, fixes

    def review_file(self, name, duration, decoded, audio, work=None):
        """All segments at once (tests)."""
        self.start_file(name, duration, audio, work, cached=decoded)
        return self.finish_file()

    def file_saved(self, out_name):
        line = f"Saved {out_name}: {self.summary().strip()}"
        self.saved.append(line)
        self.status(line)

    def close(self, summary="Finished."):
        if self.ui:
            self.ui.close("\n".join(self.saved + [summary]))

    def summary(self) -> str:
        s = self.stats
        if not s:
            return ""
        if self.mode == "blank":
            return f"  unsure parts: {s['unsure']}, blanked: {s['blanked']}"
        return (f"  unsure parts: {s['unsure']}, corrections: {s['corrected']}, "
                f"blanked: {s['blanked']}, deleted: {s['deleted']}, added to keywords: {s['added']}"
                + (f", settled by the other models: {s.get('auto', 0)} replaced, "
                   f"{s.get('confirmed', 0)} confirmed" if s.get("auto") or s.get("confirmed") else ""))
