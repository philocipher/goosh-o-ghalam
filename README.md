# Goosh-o-Ghalam (گوش و قلم)

*Ear and pen*: Persian lectures and talks → reviewed subtitles (.srt) and transcripts, offline.

The transcript is made by **Shenava Koochik v1.0** (a 114M-parameter FastConformer CTC model for Persian, runs on CPU; its name is also shown on the review page and in the window) with onnxruntime, plus Silero VAD (via sherpa-onnx) to split speech into subtitle cues. Nothing is uploaded anywhere; after the one-time model download it works offline.

## Setup (Windows): nothing to install

Copy this folder anywhere and double-click **`review.bat`** (or `transcribe.bat`). On a computer where it hasn't run yet, it first sets itself up, all inside this folder:

1. **In the black window** (about 2 minutes): a portable Python 3.12 from python.org (11 MB) goes into `python\`, and the packages this program needs (about 300 MB) are installed into it. This step needs no admin rights and leaves the rest of the computer alone. (You can also run `setup.bat` on its own; it is safe to run again.)
2. **On the review page**, with name, size and progress shown: the speech-recognition model Shenava Koochik (460 MB) and the speech detector, needed before anything can be transcribed; then, in the background while you work, the two second-opinion models **Nava** (1.6 GB) and **Vosk** (53 MB). Their progress shows in the file list ("Second-opinion models: Nava – downloading 35% of 1.6 GB…") and next to each suggestion. Two more are fetched the first time you use them: the **punctuation** model (260 MB) and **speaker detection** (34 MB).

After that it works offline. Everything (Python, packages, models in `models\`, your work in `transcripts\`) stays in this folder, so the whole folder can be copied to another computer or a USB drive and run there as is. An older `.venv` folder from a previous setup is still used if `python\` isn't there yet.

## Use

- **Drag and drop** audio/video files, or a whole folder, onto **`transcribe.bat`**.
  Each file gets its own folder inside the `transcripts` folder of this program, with the `.srt` inside (e.g. `interview.mp3` → `transcripts\interview_3f9a1c\interview.srt`). Nothing is written next to your recordings. See [Saved progress](#saved-progress).
- Or from a terminal in this folder:

```bat
python\python transcribe.py "D:\videos\talk.mp4"
python\python transcribe.py "D:\recordings" --out-dir "D:\subs"
```

Works with mp3, wav, m4a, ogg, flac, mp4, mkv, mov and most other formats (FFmpeg is bundled through PyAV, no separate install).

## Options

| Option | What it does |
|---|---|
| `--out-dir PATH` | Also copy each finished `.srt` into this folder |
| `--work-dir PATH` | Keep the per-recording folders there instead of in `transcripts` |
| `--digits` | Turn spoken numbers into digits (پانزده → ۱۵). Command line only, off by default because it also turns «نه» (“no”) into ۹ |
| `--threads N` | CPU threads (default: half your cores) |
| `--max-segment S` | Longest speech chunk sent to the model, seconds (default 15) |
| `--overwrite` | Make the .srt again even if it exists (saved progress is reused, so this is quick) |
| `--fresh` | Forget the saved progress for these files and start over |
| `--review` | Open the transcript in the browser to check the unsure parts and correct any word (see below) |
| `--review-in-terminal` | With `--review`: ask in the console window instead of the browser page |
| `--auto-replace X` | With `--review`: settle unsure parts automatically when Nava is at least this sure (default `0.6`), or `off` |
| `--alternatives M` | With `--review`: other models whose guesses are shown for unsure parts: `nava`, `vosk`, `nava,vosk` (default) or `none` |
| `--blank-uncertain` | Write `[…]` for unsure parts instead of guessing. With `--review`: blank whatever is left after you stop asking |
| `--confidence X` | Words the main model is less sure of than this (0–1) count as unsure: they're checked by Nava and Vosk and highlighted. Default `0.65` (on the 4-minute test clip, 50 unsure parts in 442 words) or the last value chosen on the review page; lower it (e.g. `0.4`) for fewer |
| `--keywords L` | Keyword lists to use, comma-separated: names of lists in the `keywords` folder (e.g. `philosophy,names`), paths to `.txt` files, or `none` (default: the ones used last) |
| `--fuzziness X` | How loosely words are matched to keywords: `0` = exact only, `1` = default, `2` = loose |

Subtitle layout (42 chars/line, 2 lines, max 7 s per cue) can be changed at the top of `transcribe.py`.

## Keywords (names and special terms)

Put names, brands and technical terms in a **keyword list**: a `.txt` file in the **`keywords`** folder, one entry per line. Keep one list per topic (`philosophy.txt`, `history.txt`, `names.txt`, …) and choose which ones to use for each recording on the review page (or with `--keywords`). After recognition, any word that looks or sounds close to a keyword (for example ض/ز/ذ/ظ or س/ص/ث mix-ups, or a word split in two) is replaced with your spelling. For a mistake that is too different, add a fixed rule: `what the model writes => what you want`. Each file reports how many corrections it made.

The `keywords` folder is yours and is kept out of the Git repository (see `.gitignore`). On a new copy the program starts it with `general.txt`, copied from `keywords.example.txt`, which explains the format; a `keywords.txt` from an earlier version is moved to `keywords\philosophy.txt`. A list of Islamic philosophy terms (from *Bidayat al-Hikma* and the *Asfar*: اصالت وجود, حرکت جوهری, واجب الوجود, …) works well, as do the corrections you add on the review page. Each word of a term may only differ a little, and short words only by a same-sounding letter, so everyday phrases such as «علت اصلی» or «در بین» are left alone.

This is a correction pass, not true biasing: the model can't be steered toward a word while it listens, so a keyword the model gets very wrong needs a fixed rule. If unrelated words start getting replaced, lower `--fuzziness` (e.g. `0.5`) or use a fixed rule for that keyword instead.

## Review unsure parts

**Double-click `review.bat`** (or run with `--review`). A review page opens in your browser (it runs only on your computer; the link is also printed in the window) with a **file list**: pick a folder (type its path, click through folders, or use the drive buttons) and click a recording. A **Before starting** dialog then asks for this recording's settings: which **keyword lists** to use (tick one or several), **Unsure below** (default 65%), whether to **check unsure words with Nava and Vosk** and **replace them automatically when Nava is at least** (default 60%) sure. Each recording remembers its own settings (shown again next time), and new recordings start from the last ones you chose. **⚙ Settings** in the header changes them for the open recording at any time: the highlighting and keyword fixes are worked out again, and lines you edited stay as they are. Each recording shows how far it has got: *not started*, *transcribed to 12:30*, *transcribed*, *subtitles saved*, and how many lines you've edited. You can also drag a recording or a folder onto `review.bat` to start with it.

The **☰ Files** button at the top opens the list again at any time, to switch to another recording. The one you leave stays exactly as it was: its edits are already saved, and its transcription carries on from where it stopped when you open it again. Reopening a recording also takes you back **to where you were** in it (press **N** for the next unsure part from there).

The header keeps only what you use all the time: ☰ Files, the counts, **↶** (undo), **🔍** (find), *Next unsure*, **⋯ More**, *Finish*, and **⏻ Close** in the top-right corner (it ends the program; everything is saved). **⋯ More** holds the rest: ⚙ Settings of this recording, 👥 Find speakers, ⤓ Save as Word & Markdown, 📂 Show its files, ⌨ Keyboard shortcuts (or press **?**) and ↻ Start this recording over.

**Closing the page closes the program** a little later (about 20 seconds; everything is saved), so it doesn't keep using the computer in the background. Reloading the page is fine. While a background queue is working (see below), the program stays open until it is done.

**↻ Start over** (in ⋯ More for the open recording, and next to each recording in the list that has saved progress) throws away everything saved for that recording (its transcript, your edits and off-topic marks, the other models' guesses and its subtitle files) and transcribes it again from the beginning. It asks first, as it can't be undone.

The **transcript appears live**, a few seconds after the program starts, one line per stretch of speech with its time, and grows as the file is transcribed (the header shows how far it has got). You can review and edit while it is still transcribing. Words the model was unsure about are highlighted, and the editor bar at the bottom opens on the first one; once you've caught up, the next unsure part opens by itself as soon as it is transcribed.

- **The player** at the top plays the whole recording at 0.5×–3× speed, and the word being spoken is underlined. The text glides along so the playing word stays about a third of the way down the screen, even while the editor bar is open. Scrolling yourself pauses following for a few seconds; **Back to the playing word** brings it back at once. The seek bar fills in as the file is transcribed, like a video loading, and you can jump anywhere in the transcribed part; if playback catches up with the transcription it waits and then carries on. Click a line's time to play from there. Keys, when you're not typing: **Space**/**K** play-pause, **←/→** 5 s back/forward, **J/L** 10 s, **Shift+<** / **Shift+>** slower/faster.
- **The seek bar shows where the work is**: a yellow tick for each unsure part, blue for automatic changes, a hatched band for off-topic stretches, and (once found) a coloured strip for who speaks when.
- **Click any word** to hear it and edit it; **Shift+click** another word in the same line to select several. The clip includes 0.2 s either side, so the start and end of the words aren't cut off. Selecting also moves the player at the top to just before those words, so **Space** plays on from there.
- A word **you changed** shows what the model had heard (hover it, or in the editor bar) with **Put back what it heard**.
- **Undo** (**↶** or **Ctrl+Z**) takes back your last change, mark or replace-all, up to 100 steps (for the open recording, until the program closes). A word added to a keyword list stays in the list.
- **Enter** saves what's in the box. On a highlighted part, Enter with the text unchanged keeps the model's guess, and either way it moves on to the next highlighted part. **N** jumps to the next one, **Esc** closes the bar, **Ctrl+Space** replays; tick *slower* for 75% speed.
- **Ctrl+B** / *Leave blank* writes `[…]` in the subtitles; *Delete words* removes them.
- **Replace all N in this recording** appears when the selected words occur more than once: it puts what you typed in place of every one of them (whole words; spelling variants like ي/ی and half-spaces count as the same). Unlike a keyword rule, it changes only this recording.
- Before saving a correction you can also add it to the first ticked keyword list, **as a keyword** (what you typed) or **as a rule** (always change what was heard into what you typed). It's applied at once to every line you haven't edited, and to future files. Rules aren't offered for short single words like «سر», which would be changed everywhere.
- **Finish & save subtitles** writes the .srt, and the transcript as a document (see below). If some parts are still highlighted, it asks whether to keep them as heard or leave them blank. You can press it before transcription is done; the rest of the file then gets the same choice and the .srt is written when it completes. Your edits are kept if you reload the page.

### Marking off-topic parts

Greetings, connection trouble, small talk and the like can be marked, word by word, so they're left out of a lesson-only copy of the output.

- **Select the words**: drag across them with the mouse (across lines too), or click the first word and **Shift+click** the last. A single click selects one word.
- Press **Mark off-topic** in the editor bar, or **M** (**Ctrl+M** while the cursor is in the text box). Marked words are greyed and struck through, with a small *⊘ off-topic* where the stretch starts.
- To take words back out, select them and press **Unmark** or **Shift+M**; the rest of the stretch stays marked.
- **hide off-topic** (next to the player) hides the marked words and skips them while playing, so you can read and listen to just the lesson.
- Unsure words inside marked stretches aren't counted or visited by *Next unsure*: they'll be left out anyway.

When you press Finish, besides `<name>.srt` (everything) the folder gets:

| File | What it holds |
|---|---|
| `<name>.lesson.srt` | the subtitles without the marked words (original timings) |
| `<name>.lesson.txt` | the lesson as plain text, a new paragraph wherever something was cut out |
| `<name>.offtopic.csv` | start, end and words of each marked stretch (opens in Excel; handy for cutting them out of the audio or video too) |

Marks are saved with your edits, so they survive switching recordings and closing the program.

### Punctuation

The main model writes words only, so `.` `،` `؟` `:` are added by a second model once a recording is transcribed (about 20 seconds per hour of audio; always on, nothing to set). It never changes words, only adds marks after them, so the word timings stay right.

The model is [Horizon-Labs/punctuation-restoration-small](https://huggingface.co/Horizon-Labs/punctuation-restoration-small) (88 languages including Persian, Apache-2.0, 260 MB). On its Persian test it gets 82% of marks right on clean text and 57% on speech-recognition output; lectures with many Arabic quotations are harder. It is tuned here to add somewhat more marks than it would by itself, which read more naturally in lectures. A Persian-only model (PersianPunc, ParsBERT) scores higher but needs a Hugging Face login to download, so it can't be set up automatically.

### Speakers

**👥 Find speakers…** (in ⋯ More) works out who speaks when, from the voices: about 7 seconds per minute of audio (11 minutes for a 95-minute lecture), in the background while you keep working. Say how many people speak if you know: it works best that way; left empty, it decides by itself. Each line then gets the name of its speaker where the speaker changes, and the seek bar shows who speaks when in colour. **Click a name** to rename that speaker everywhere (e.g. *Speaker 1* → «استاد»). Names appear in the Word and Markdown files.

It uses [pyannote segmentation 3.0 and the 3D-Speaker CAM++ voice-print model](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html) through sherpa-onnx (34 MB). Clubhouse audio is compressed and voices on phone microphones sound alike, so check the result; giving the number of speakers helps most.

### Find and replace

**🔍** (or **Ctrl+F**) finds words or phrases, in **this recording** (click a result to select it) or in **all recordings** in the `transcripts` folder (with your edits; click a result to open that recording at that line). Spelling variants (ي/ی, ك/ک, half-spaces, short vowels) count as the same. In this recording, **Replace all** replaces every occurrence as whole words (Ctrl+Z undoes it).

### A whole folder in the background

In the file list, **⏩ Transcribe all here in the background** queues every recording in that folder that isn't transcribed yet. They are transcribed one after another while you review something else; the one you have open always comes first (the queue pauses while it is being transcribed), and opening a queued recording takes it out of the queue and carries on from where the queue got. The header shows *Queue: working 43% · 2 more* (click it for the file list); **Stop the queue** keeps what's done. You can close the page and leave it running overnight: the program closes itself when the queue is done. Punctuation, the other models and speakers run when you open each recording.

### Word and Markdown

With the subtitles, Finish also writes `<name>.docx` and `<name>.md`: the transcript in paragraphs, each with its start time, with punctuation, the speaker's name where it changes (if speakers were found), and without the off-topic stretches. A new paragraph starts at a pause of 8 seconds or more, a change of speaker, a cut, or a sentence end after about 180 words. **⤓ Save as Word & Markdown** in ⋯ More writes them at any time, as the transcript is now.

### Other models' guesses

Under the text box, the editor bar also shows what two other recognisers hear in the same clip, with the words that line up with your selection as a button (click it to put it in the box; a green outline means it agrees with the current text), followed by that model's **confidence** in those words (red below 50%) and the full text of what it heard in the clip. Hover a model's name (here, in the header, or *heard by* in the editor bar) for what it is and its size, and click it to open the model's page. Hover the button for the confidence of each word, or a word of the full text for its confidence. Both models' confidences work like the main model's: the probability of each word (Nava: its least certain piece; Vosk: its own per-word score), and for several words the lowest one. They're each model's own measure, so compare them within a model rather than across models.

- **Nava**: OpenAI's Whisper large-v3-turbo, fine-tuned on 579 hours of Persian by its author ([Nimruz/Nava](https://huggingface.co/Nimruz/Nava), MIT licence, marked as a preview). It makes different mistakes from the main model, so it is the more useful second opinion. 1.6 GB.
- **Vosk**: a different kind of recogniser that only writes words from its dictionary ([vosk-model-small-fa-0.42](https://alphacephei.com/vosk/models), Apache 2.0). Fast but less accurate. 53 MB.

**Settled automatically.** As each guess comes in, an unsure part is settled without asking you when the other models are sure enough:

- **Nava** is at least 60% sure of other words, *and surer of them than Shenava was of its own* → they replace the unsure ones, shown in **blue with a dashed underline**;
- **Nava** is at least 60% sure of the *same* words → they're **confirmed** (no longer unsure, left as they are);
- **Nava and Vosk** heard the same words, both at least 50% sure → same as above.

Vosk alone never decides: its scores are too optimistic (on the test clip it gave 100% to «برای», «جذاب از» and «داریم به نام», all wrong), so it only counts when it heard the same words as Nava. The words must also line up with ours on both sides and not be much longer than the unsure part, and the part open in the editor bar is never touched. They are saved like your edits and counted in the summary.

To check them, click the count in the header (*6 changed · 4 confirmed automatically ▾*): a list shows every one with its time, *what was heard ← what it became*, and who decided how surely. Click a row to open it in the text (the player moves there too), or **Undo** to put back what the main model had heard. While one is open, the editor bar shows **‹ 2 of 10 ›** to step through them, and **Put back the original**. Hovering a changed word in the text also shows what was there before.

On the 4-minute test clip this settled 10 of 27 unsure parts; some changes were clearly right («اطلابه» → «اطلاقه», «اه» → «اعمه»), some wrong («داری؟» → «داریم بنابرا»), so it's worth going through the list. `--auto-replace 0.8` is stricter, `--auto-replace off` turns it off.

They are downloaded into `models/alt` the first time you review, and only run on the short clips around unsure parts (and on words you click): the part you have open first, then the following unsure parts in the background. Expect a few seconds per clip for Nava on a CPU. Turn them off with `--alternatives none`, or keep one with `--alternatives vosk`.

Keep the black window open while you review. Ctrl+C there finishes the file as it is. To answer question by question in the window instead of the browser, add `--review-in-terminal` (the Windows console may show Persian letters unjoined).

To leave the unsure parts blank without being asked, use `--blank-uncertain`. If too much is highlighted, lower `--confidence` (default `0.65`, e.g. `0.4`); to catch more mistakes, raise it. Confidence comes from the same model run, so these modes are no slower than a normal transcription (apart from the time you spend answering).

## Saved progress

Everything made for a recording is kept in its own folder inside `transcripts` (next to this program, or `--work-dir`), named after the recording plus a short tag, so two recordings with the same name in different places don't mix:

| File | What it holds |
|---|---|
| `<name>.srt` | the subtitles |
| `<name>.docx`, `<name>.md` | the transcript as a document |
| `segments.jsonl` | every transcribed line, saved the moment it's ready |
| `review.json` | your edits, off-topic marks and where you were, saved after each change |
| `alternatives.json` | what Nava and Vosk heard in each clip |
| `punctuation.json` | the punctuation marks (the model's and yours) |
| `speakers.json` | who speaks when, and the names you gave |
| `source.txt` | where the recording is (so search results can open it) |

So nothing is lost if the window is closed, the computer restarts or the program crashes: run it again on the same file and it carries on. Transcription continues from the last saved line (lines right after that point can differ by a comma, because speech detection restarts there), the review page reopens with your edits, and the other models' guesses appear at once. With `review.bat` you can also reopen a finished file to change more; your earlier edits are still there.

The saved lines are used only for the same file with the same settings; if the file changes, it starts over by itself. To start over on purpose, use **↻ Start over** on the review page, add `--fresh`, or delete the folder. Progress folders that an earlier version left next to a recording (`<name>_transcript`) are moved into `transcripts` automatically. Changing keyword lists or the unsure threshold doesn't need a fresh start: they're applied to the saved lines.

Video players only pick up subtitles automatically when the `.srt` sits next to the video with the same name: copy it there, or use `--out-dir` to have a copy made.

## Notes

- Speed: roughly 20× faster than real time on a typical laptop CPU (a 1-hour file takes a few minutes).
- The main model outputs no punctuation (it's added by a second model, see [Punctuation](#punctuation)). It is strongest on clear speech; noisy or overlapping speech will have more errors.
- Graphics card: everything runs on the CPU. Running the main model on the graphics card (DirectML) was tried on an Intel Iris Xe laptop: on real speech it was about 2× *slower* than the CPU (the short, varying-length pieces of speech don't suit it), so it isn't offered. The Nava model could only use an NVIDIA card (CUDA).
- Model: [Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx) (Apache-2.0).
