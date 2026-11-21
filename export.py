"""
The transcript as a document to read: <name>.md (Markdown) and <name>.docx (Word), next to
the .srt. Paragraphs with their start time, a speaker's name where the speaker changes (if
speakers were found), and the off-topic stretches left out (if any were marked).

The .docx is written directly (it is a zip of a few XML files), so no extra package is needed;
paragraphs are right-to-left.
"""

import zipfile
from datetime import datetime, timezone
from xml.sax.saxutils import escape

PAUSE_BREAK = 8.0     # seconds of silence that start a new paragraph
MAX_WORDS = 180       # a paragraph this long breaks at the next sentence end
_SENTENCE_END = (".", "؟", "?", "!")


def _clock(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def paragraphs(decoded, marks=(), turns=(), names=None, itn=None):
    """
    decoded: [(start, dur, [(word, time, ...)])] (words may end with punctuation).
    -> [{"t": start sec, "speaker": name or None, "text": str}]
    """
    from speakers import speaker_of
    names = names or {}

    def marked(t):
        return any(m["a"] <= t < m["b"] for m in marks)

    def name_of(spk):
        return None if spk is None else (names.get(str(spk)) or f"Speaker {spk + 1}")

    out, cur, cur_t, cur_spk, last_end, cut = [], [], None, None, None, False

    shown = [object()]     # the speaker named last (a name is written only when it changes)

    def flush():
        if cur:
            text = " ".join(cur)
            name = name_of(cur_spk)
            out.append({"t": cur_t, "speaker": name if name != shown[0] else None,
                        "text": itn(text) if itn else text})
            shown[0] = name
        cur.clear()

    for start, dur, words in decoded:
        kept = [w for w in words if not marked(w[1])]
        if len(kept) < len(words):
            cut = True
        if not kept:
            continue
        spk = speaker_of(turns, start, start + dur) if turns else None
        new_para = (not cur or cut or (last_end is not None and start - last_end > PAUSE_BREAK)
                    or (turns and spk is not None and spk != cur_spk)
                    or (len(cur) >= MAX_WORDS and cur[-1].endswith(_SENTENCE_END)))
        if new_para:
            flush()
            cur_t, cur_spk = kept[0][1], spk if turns else None
        cur.extend(w[0] for w in kept)
        cut = len(kept) < len(words) and words[-1] is not kept[-1]
        last_end = start + dur
    flush()
    return out


def write_markdown(path, title, paras, note=""):
    lines = [f"# {title}", ""]
    if note:
        lines += [f"*{note}*", ""]
    for p in paras:
        head = f"**{p['speaker']}** " if p["speaker"] else ""
        lines += [f"{head}`{_clock(p['t'])}`", "", p["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- .docx
_CT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""
_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""
_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults>
<w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="B Nazanin"/><w:sz w:val="24"/><w:szCs w:val="28"/><w:lang w:bidi="fa-IR"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:bidi/><w:spacing w:after="160" w:line="300" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr></w:pPrDefault>
</w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:bCs/><w:sz w:val="36"/><w:szCs w:val="40"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Meta"><w:name w:val="Meta"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="40"/></w:pPr><w:rPr><w:color w:val="6B6B70"/><w:sz w:val="18"/><w:szCs w:val="20"/></w:rPr></w:style>
</w:styles>"""


def _run(text, bold=False, rtl=True):
    rpr = ("<w:b/><w:bCs/>" if bold else "") + ("<w:rtl/>" if rtl else "")
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _para(runs, style=None):
    ppr = f'<w:pStyle w:val="{style}"/>' if style else ""
    return f"<w:p><w:pPr>{ppr}<w:bidi/></w:pPr>{''.join(runs)}</w:p>"


def write_docx(path, title, paras, note=""):
    body = [_para([_run(title, rtl=any("؀" <= c <= "ۿ" for c in title))], "Title")]
    if note:
        body.append(_para([_run(note, rtl=False)], "Meta"))
    for p in paras:
        meta = [_run(_clock(p["t"]), rtl=False)]
        if p["speaker"]:
            meta = [_run(p["speaker"], bold=True), _run("  ", rtl=False)] + meta
        body.append(_para(meta, "Meta"))
        body.append(_para([_run(p["text"])]))
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
           + "".join(body) +
           '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
           '<w:bidi/></w:sectPr></w:body></w:document>')
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{escape(title)}</dc:title>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>')
    tmp = path.with_name(path.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        z.writestr("word/styles.xml", _STYLES)
        z.writestr("word/document.xml", doc)
        z.writestr("docProps/core.xml", core)
    tmp.replace(path)


def write_documents(stem, title, decoded, marks=(), turns=(), names=None, itn=None):
    """Write <stem>.md and <stem>.docx; returns their paths."""
    paras = paragraphs(decoded, marks, turns, names, itn)
    note = "Off-topic parts are left out." if marks else ""
    md = stem.with_name(stem.name + ".md")
    docx = stem.with_name(stem.name + ".docx")
    write_markdown(md, title, paras, note)
    write_docx(docx, title, paras, note)
    return [md, docx]
