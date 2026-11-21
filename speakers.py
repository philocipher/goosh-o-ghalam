"""
Who speaks when (speaker diarization), offline, with sherpa-onnx (already installed):
  pyannote segmentation 3.0  finds stretches of speech by one voice (6 MB)
  3D-Speaker CAM++           turns each stretch into a voice print, then similar ones are
                             grouped into speakers (28 MB; 2.7x faster than ERes2Net here,
                             about 7 s per minute of audio on a laptop CPU)
Downloaded on first use into models/speakers. The result is a list of turns
(start sec, end sec, speaker number), saved per recording in speakers.json; names given
on the review page are saved there too.
"""

import tarfile
import time
from pathlib import Path

ABOUT = "pyannote segmentation 3.0 + 3D-Speaker CAM++ voice prints, via sherpa-onnx · 34 MB"
URL = "https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html"
_SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
            "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EMB = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
_EMB_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{EMB}"
SEG = "segmentation.onnx"
# How different two voice prints may be and still count as one speaker, when the number of
# speakers isn't given (larger: fewer speakers). On a Clubhouse lecture, sherpa-onnx's usual
# 0.5 made 20 "speakers" in 10 minutes; 0.9-1.0 gives a handful.
THRESHOLD = 0.9


def folder(model_dir: Path) -> Path:
    return Path(model_dir) / "speakers"


def missing(model_dir: Path):
    d = folder(model_dir)
    return [n for n in (SEG, EMB) if not ((d / n).exists() and (d / n).stat().st_size > 0)]


def download(model_dir: Path, status=None):
    import workdir
    d = folder(model_dir)
    todo = missing(model_dir)
    last = [0.0]

    def progress(what, k):
        def cb(done, total):
            if status and (time.time() - last[0] > 0.3 or done == total):
                last[0] = time.time()
                frac = done / total if total else None
                status(f"Setting up speaker detection ({k} of {len(todo)}): downloading {what}"
                       + (f"… {frac:.0%}" if frac is not None else ""), frac)
        return cb

    for k, name in enumerate(todo, 1):
        if name == SEG:
            arch = d / "segmentation.tar.bz2"
            workdir.fetch(_SEG_URL, arch, progress("the speech-turn finder (6 MB)", k))
            with tarfile.open(arch) as tf:   # only the model file is taken from the archive
                member = next(m for m in tf.getmembers()
                              if m.isfile() and Path(m.name).name == "model.onnx")
                src = tf.extractfile(member)
                tmp = d / (SEG + ".part")
                tmp.write_bytes(src.read())
                tmp.replace(d / SEG)
            arch.unlink(missing_ok=True)
        else:
            workdir.fetch(_EMB_URL, d / EMB, progress("the voice-print model (28 MB)", k))


def find_speakers(audio, model_dir: Path, num_speakers=None, threads=4, progress=None):
    """
    audio: 16 kHz mono float32. -> [(start, end, speaker 0..n-1)] in time order, speakers
    numbered in the order they first speak. progress(fraction) while it works.
    """
    import sherpa_onnx
    d = folder(model_dir)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(d / SEG)),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(d / EMB), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=int(num_speakers) if num_speakers else -1, threshold=THRESHOLD),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("speaker detection models are incomplete")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def cb(done, total):
        if progress and total:
            progress(done / total)
        return 0

    result = sd.process(audio, callback=cb).sort_by_start_time()
    turns = [(float(r.start), float(r.end), int(r.speaker)) for r in result]
    order = {}
    for _, _, s in turns:            # Speaker 1 is whoever speaks first
        order.setdefault(s, len(order))
    return [(a, b, order[s]) for a, b, s in turns]


def speaker_of(turns, a, b):
    """The speaker who talks most in [a, b) (None if nobody)."""
    best, best_t = None, 0.0
    share = {}
    for s0, s1, spk in turns:
        if s1 <= a:
            continue
        if s0 >= b:
            break
        share[spk] = share.get(spk, 0.0) + min(b, s1) - max(a, s0)
    for spk, t in share.items():
        if t > best_t:
            best, best_t = spk, t
    return best
