"""
One folder per input file, holding everything made for it, so an interrupted run carries
on where it stopped instead of starting over. They all live in one place (by default the
"transcripts" folder next to this program), not next to the recordings:

    transcripts/<name>_<tag>/        (tag: from the recording's full path, so two
                                      recordings with the same name don't mix)
        <name>.srt          the subtitles (written at the end)
        segments.jsonl      every transcribed line, saved as soon as it's ready
        review.json         edits made on the review page
        alternatives.json   what the other models (Nava, Vosk) heard in each clip

segments.jsonl starts with a line describing the input file and the settings; if the file
or the settings change, the saved lines no longer apply and the folder starts over.
"""

import json
import os
import threading
from pathlib import Path

FORMAT = 1
CACHE_FILES = ("segments.jsonl", "review.json", "alternatives.json", "punctuation.json", "speakers.json")
OUTPUT_EXTS = (".srt", ".lesson.srt", ".lesson.txt", ".offtopic.csv", ".md", ".docx")

_FA = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "ٱ": "ا",
                     "‌": " ", "ـ": None, **{chr(c): None for c in range(0x064B, 0x0653)},
                     **{c: " " for c in ".,;:!?؟،؛«»()[]\"'…"}})


def normalize_fa(text: str) -> str:
    """For comparing Persian text: one spelling of ی ک ا, no short vowels, a half-space as a space."""
    return " ".join(text.translate(_FA).lower().split())


DEFAULT_ROOT = Path(__file__).resolve().parent / "transcripts"


def _home_for(media: Path, root: Path | None = None) -> Path:
    """Where a recording's folder belongs."""
    import hashlib
    tag = hashlib.sha1(str(media.resolve()).lower().encode("utf-8")).hexdigest()[:6]
    return (root or DEFAULT_ROOT) / f"{media.stem}_{tag}"


def folder_for(media: Path, root: Path | None = None) -> Path:
    """Where a recording's folder is (an earlier version's one next to it, if it couldn't be moved)."""
    new = _home_for(media, root)
    old = media.parent / f"{media.stem}_transcript"
    return old if (old.is_dir() and not new.exists()) else new


def adopt_old_folder(media: Path, root: Path | None = None) -> None:
    """
    Earlier versions kept the progress folder next to the recording (<name>_transcript);
    move it to its new place, if there's nothing there yet.
    """
    import shutil
    old = media.parent / f"{media.stem}_transcript"
    new = _home_for(media, root)
    if old.is_dir() and not new.exists():
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
            print(f"  moved saved progress from {old} to {new}")
        except OSError as e:
            print(f"  couldn't move {old} ({e}); it is used from there for now")


# Settings chosen per recording on the review page ("Before starting"), and the last ones
# used, which new recordings start from.


def clean_settings(raw, fallback: dict, lists=None) -> dict:
    """Settings from the page or a file, checked: anything missing or odd falls back."""
    out = dict(fallback)
    if not isinstance(raw, dict):
        return out
    kw = raw.get("keywords")
    if isinstance(kw, list):
        kw = [str(k) for k in kw if isinstance(k, str)]
        out["keywords"] = [k for k in kw if lists is None or k in lists]
    for key in ("confidence", "auto_replace"):
        v = raw.get(key, out.get(key))
        if v is None and key == "auto_replace":
            out[key] = None
        elif isinstance(v, (int, float)) and 0 < float(v) <= 1:
            out[key] = round(float(v), 3)
    for key in ("alternatives",):
        if isinstance(raw.get(key), bool):
            out[key] = raw[key]
    return out


def load_defaults(root: Path | None = None):
    try:
        return json.loads(((root or DEFAULT_ROOT) / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_defaults(root: Path | None, settings: dict) -> None:
    try:
        (root or DEFAULT_ROOT).mkdir(parents=True, exist_ok=True)
        _write_json((root or DEFAULT_ROOT) / "settings.json", settings)
    except OSError:
        pass


def last_folder(root: Path | None = None) -> Path:
    """The folder a recording was last picked from on the review page (or the home folder)."""
    try:
        p = Path(((root or DEFAULT_ROOT) / "last_folder.txt").read_text(encoding="utf-8").strip())
        if p.is_dir():
            return p
    except OSError:
        pass
    return Path.home()


def remember_folder(root: Path | None, folder: Path) -> None:
    try:
        (root or DEFAULT_ROOT).mkdir(parents=True, exist_ok=True)
        ((root or DEFAULT_ROOT) / "last_folder.txt").write_text(str(folder), encoding="utf-8")
    except OSError:
        pass


def status(media: Path, root: Path | None = None) -> dict:
    """
    How far a recording has got, cheaply (for the file list):
      {"state": "new" | "partial" | "transcribed" | "done", "until": seconds transcribed,
       "edits": lines edited in review}
    """
    folder = folder_for(media, root)
    out = {"state": "new", "until": 0.0, "edits": 0}
    remember_source(folder, media)     # (links folders made before source.txt existed)
    seg = folder / "segments.jsonl"
    if seg.exists():
        try:
            with open(seg, "rb") as f:
                f.seek(max(0, seg.stat().st_size - 8192))
                tail = f.read().decode("utf-8", "ignore").splitlines()
            lines = [json.loads(x) for x in tail[1:] if x.strip().startswith("{")][-2:]
        except (OSError, ValueError):
            lines = []
        if lines and lines[-1].get("done"):
            out["state"] = "transcribed"
        elif lines:
            last = lines[-1]
            out["state"] = "partial"
            out["until"] = round(last.get("start", 0) + last.get("dur", 0), 1)
    try:
        review = json.loads((folder / "review.json").read_text(encoding="utf-8"))
        out["edits"] = len(review.get("edited", {}))
    except (OSError, ValueError):
        pass
    if (folder / f"{media.stem}.srt").exists():
        out["state"] = "done"
    return out


def remember_source(folder: Path, media: Path) -> None:
    """Note which recording a folder belongs to (search results open it from there)."""
    f = folder / "source.txt"
    try:
        p = str(media.resolve())
        if folder.is_dir() and (not f.exists() or f.read_text(encoding="utf-8").strip() != p):
            f.write_text(p, encoding="utf-8")
    except OSError:
        pass


def source_of(folder: Path):
    try:
        return Path((folder / "source.txt").read_text(encoding="utf-8").strip())
    except OSError:
        return None


_lines_cache = {}   # folder -> (file times, lines)


def transcript_lines(folder: Path):
    """[(start sec, text)] of a recording's transcript as saved, with the review edits."""
    seg, rev = folder / "segments.jsonl", folder / "review.json"
    try:
        key = tuple(p.stat().st_mtime if p.exists() else 0 for p in (seg, rev))
    except OSError:
        return []
    hit = _lines_cache.get(folder)
    if hit and hit[0] == key:
        return hit[1]
    lines = []
    try:
        with open(seg, encoding="utf-8") as f:
            next(f, None)                    # the header
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    break
                if "words" in d:
                    lines.append([d["start"], [w[0] for w in d["words"]]])
    except OSError:
        return []
    try:
        edited = json.loads(rev.read_text(encoding="utf-8")).get("edited", {})
        for k, words in edited.items():
            if 0 <= int(k) < len(lines):
                lines[int(k)][1] = [w[0] for w in words]
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    out = [(start, " ".join(words)) for start, words in lines]
    _lines_cache[folder] = (key, out)
    return out


def search(root: Path | None, query: str, limit: int = 300):
    """
    Lines of every saved transcript that contain `query` (spelling variants and half-spaces
    alike): {"hits": [{"file", "path", "line", "t", "text"}], "files": n, "more": bool}.
    """
    q = normalize_fa(query)
    out = {"hits": [], "files": 0, "more": False}
    if len(q) < 2:
        return out
    root = root or DEFAULT_ROOT
    try:
        folders = sorted((p for p in root.iterdir() if (p / "segments.jsonl").exists()),
                         key=lambda p: p.name.lower())
    except OSError:
        return out
    for folder in folders:
        src = source_of(folder)
        found = False
        for k, (start, text) in enumerate(transcript_lines(folder)):
            if q in normalize_fa(text):
                if len(out["hits"]) >= limit:
                    out["more"] = True
                    return out
                found = True
                out["hits"].append({"file": src.name if src else folder.name,
                                    "path": str(src) if src and src.exists() else None,
                                    "line": k, "t": start, "text": text})
        out["files"] += found
    return out


def fetch(url: str, dest: Path, progress=None) -> None:
    """
    Download url to dest (via dest.part, so a broken download is never taken for a whole
    one). progress(bytes done, bytes total) is called as it goes.
    """
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "goosh-o-ghalam"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if progress:
                progress(done, total)
    os.replace(tmp, dest)


def _write_json(path: Path, data) -> None:
    """Write atomically, so a crash never leaves half a file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class WorkDir:
    def __init__(self, folder: Path, media: Path, settings: dict):
        self.dir = folder
        self.media = media
        st = media.stat()
        self.header = {"format": FORMAT, "source": media.name, "size": st.st_size,
                       "mtime": int(st.st_mtime), **settings}
        self.srt = folder / f"{media.stem}.srt"
        self._segments = folder / "segments.jsonl"
        self._review = folder / "review.json"
        self._alts = folder / "alternatives.json"
        self._out = None
        self._lock = threading.Lock()
        self._alt_data = None

    # -- whole folder ----------------------------------------------------------
    def create(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        remember_source(self.dir, self.media)

    def reset(self, outputs: bool = False):
        """Forget everything saved; with outputs, the subtitle files too (a start from scratch)."""
        for name in CACHE_FILES:
            (self.dir / name).unlink(missing_ok=True)
        if outputs:
            for ext in OUTPUT_EXTS:
                (self.dir / f"{self.media.stem}{ext}").unlink(missing_ok=True)
        self._alt_data = None

    def load_json(self, name):
        try:
            return json.loads((self.dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_json(self, name, data):
        with self._lock:
            self.create()
            _write_json(self.dir / name, data)

    # -- transcribed lines -------------------------------------------------------
    def load_segments(self):
        """-> ([(start, dur, [(word, time, conf)])], finished?) saved for this input."""
        if not self._segments.exists():
            return [], False
        segs, done = [], False
        with open(self._segments, encoding="utf-8") as f:
            lines = f.read().splitlines()
        try:
            if json.loads(lines[0]) != self.header:
                return None, False          # another file or other settings
        except (IndexError, ValueError):
            return None, False
        for line in lines[1:]:
            try:
                d = json.loads(line)
            except ValueError:
                break                       # cut short by a crash: keep what came before
            if d.get("done"):
                done = True
                break
            segs.append((d["start"], d["dur"], [tuple(w) for w in d["words"]]))
        return segs, done

    def open_segments(self, resume: bool):
        self.create()
        if resume:
            self._out = open(self._segments, "a", encoding="utf-8")
        else:
            self._out = open(self._segments, "w", encoding="utf-8")
            self._out.write(json.dumps(self.header, ensure_ascii=False) + "\n")
            self._out.flush()

    def add_segment(self, start, dur, words):
        self._out.write(json.dumps({"start": round(start, 3), "dur": round(dur, 3),
                                    "words": [[w[0], round(float(w[1]), 3),
                                               round(float(w[2]) if len(w) > 2 else 1.0, 4)]
                                              for w in words]},
                                   ensure_ascii=False) + "\n")
        self._out.flush()

    def finish_segments(self):
        if self._out:
            self._out.write('{"done": true}\n')
            self._out.close()
            self._out = None

    def close(self):
        if self._out:
            self._out.close()
            self._out = None

    # -- review edits --------------------------------------------------------------
    def load_settings(self):
        try:
            return json.loads((self.dir / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_settings(self, settings: dict) -> None:
        with self._lock:
            self.create()
            _write_json(self.dir / "settings.json", settings)

    def load_review(self):
        try:
            return json.loads(self._review.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_review(self, state):
        with self._lock:
            self.create()
            _write_json(self._review, state)

    # -- other models' guesses -----------------------------------------------------------
    def load_alternatives(self):
        """-> {(start, end): {model name: {"words", "conf"}} (older saves: text)}"""
        try:
            raw = json.loads(self._alts.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        self._alt_data = raw
        out = {}
        for k, v in raw.items():
            try:
                a, b = (float(x) for x in k.split("-"))
            except ValueError:
                continue
            out[(a, b)] = v
        return out

    def save_alternative(self, a, b, name, result):
        with self._lock:
            if self._alt_data is None:
                self.load_alternatives()
            self._alt_data.setdefault(f"{a:.2f}-{b:.2f}", {})[name] = result
            self.create()
            _write_json(self._alts, self._alt_data)
