"""
Punctuation for the transcript: the main model writes words only. This adds . ، ؟ : after
words, never changing the words themselves (so their times stay right).

Model: Horizon-Labs/punctuation-restoration-small (mmBERT-small, 88 languages incl. Persian,
Apache-2.0), its int8 ONNX file (256 MB), run with onnxruntime and the tokenizers package
(both installed already). Downloaded on first use into models/punctuation.
"""

import re
import threading
from pathlib import Path

import numpy as np

TITLE = "Punctuation Restoration (small)"
ABOUT = "Adds . ، ؟ : after words, in 88 languages incl. Persian · 140M parameters (256 MB)"
URL = "https://huggingface.co/Horizon-Labs/punctuation-restoration-small"
_HF = "https://huggingface.co/Horizon-Labs/punctuation-restoration-small/resolve/main"
FILES = {"model_quantized.onnx": f"{_HF}/onnx/model_quantized.onnx",
         "tokenizer.json": f"{_HF}/tokenizer.json"}
SIZE = "256 MB"

CHUNK = 400      # tokens per pass (the model reads up to 8192, but shorter is faster on a CPU)
OVERLAP = 60     # tokens shared by neighbouring passes (the first pass's guesses near its end are dropped)
# Lecture speech has no capital letters or pauses in the text to go by, and the model then
# leaves out many sentence ends. Making "no mark" this much less likely (in logits) roughly
# doubles the marks; checked on a Clubhouse lecture: 1.0 reads naturally, 2.0 cuts too often.
NO_MARK_BIAS = 1.0
# Persian marks for the model's labels ("-" a free-standing dash is rarely right: left out)
MARK = {".": ".", ",": "،", "?": "؟", ":": ":"}
_ENDS_WITH_MARK = re.compile(r"[.,;:!?؟،؛…»«)\]]$")


def folder(model_dir: Path) -> Path:
    return Path(model_dir) / "punctuation"


def missing(model_dir: Path):
    d = folder(model_dir)
    return [n for n in FILES if not ((d / n).exists() and (d / n).stat().st_size > 0)]


def download(model_dir: Path, status=None):
    """status(text, fraction) while downloading (first use only)."""
    import time
    import workdir
    todo = missing(model_dir)
    for k, name in enumerate(todo, 1):
        last = [0.0]

        def progress(done, total, name=name, k=k):
            if status and (time.time() - last[0] > 0.3 or done == total):
                last[0] = time.time()
                frac = done / total if total else None
                status(f"Setting up punctuation ({k} of {len(todo)}): downloading {name} "
                       f"({total / 1e6:.0f} MB)" + (f"… {frac:.0%}" if frac is not None else ""), frac)
        workdir.fetch(FILES[name], folder(model_dir) / name, progress)


class Punctuator:
    _lock = threading.Lock()

    def __init__(self, model_dir: Path, threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = folder(model_dir)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(d / "model_quantized.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.cls = self.tok.token_to_id("<bos>")
        self.sep = self.tok.token_to_id("<eos>")
        self.labels = ["0", ".", ",", "?", "-", ":"]   # (config.json id2label)

    def marks(self, words, progress=None):
        """
        words: [str] in reading order -> [mark or ""] for each word (the mark goes after it).
        progress(fraction) is called after each pass.
        """
        spans, pos, parts = [], 0, []
        for w in words:
            parts.append(w)
            spans.append((pos, pos + len(w)))
            pos += len(w) + 1
        text = " ".join(parts)
        if not text.strip():
            return [""] * len(words)
        enc = self.tok.encode(text, add_special_tokens=False)
        ids, offs = enc.ids, enc.offsets
        end_label = {}                       # character end of a token -> label
        start = 0
        while start < len(ids):
            end = min(len(ids), start + CHUNK)
            x = np.array([[self.cls] + ids[start:end] + [self.sep]], dtype=np.int64)
            feed = {"input_ids": x}
            if "attention_mask" in self.inputs:
                feed["attention_mask"] = np.ones_like(x)
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(x)
            with self._lock:
                logits = self.sess.run(None, feed)[0][0, 1:-1]
            logits[:, 0] -= NO_MARK_BIAS
            pred = logits.argmax(-1)
            keep_to = end if end == len(ids) else end - OVERLAP
            for k in range(start, keep_to):
                s, e = offs[k]
                if e > s:
                    end_label[e] = self.labels[int(pred[k - start])]
            if progress:
                progress(keep_to / len(ids))
            if end == len(ids):
                break
            start = keep_to
        out = []
        for w, (s, e) in zip(words, spans):
            lab = end_label.get(e, "0")
            out.append("" if lab not in MARK or _ENDS_WITH_MARK.search(w) else MARK[lab])
        return out
