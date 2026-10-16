# Kweli Lens AI

**Evidence-based fact-checking support for journalists in East Africa**, built on
[Loki (OpenFactVerification)](https://github.com/Libr-AI/OpenFactVerification) by LibrAI.

Kweli Lens takes suspicious content (text, a web link, an image, audio, or video), extracts the factual
claims, searches the web for evidence, and produces a report showing, for every claim, which sources support it,
which contradict it, and which only give context. Every source is linked. Verdicts come from fixed, documented
rules, not from the AI's opinion. **A journalist must always review the evidence and make the final editorial decision.**

> Original Loki README: [docs/loki_original_README.md](docs/loki_original_README.md)

---

## Contents
1. [What it does](#what-it-does)
2. [Feature status (honest)](#feature-status)
3. [Requirements](#requirements)
4. [Windows setup](#windows-setup-step-by-step)
5. [API keys](#api-keys-and-configuration)
6. [Audio and video transcription (local, free)](#audio-and-video-transcription-local-free)
7. [Running the app](#running-the-web-app)
8. [Example inputs](#example-inputs)
9. [Command line](#command-line-use)
10. [Tests](#running-the-tests)
11. [How verdicts work](#how-verdicts-work)
12. [Costs](#api-costs-and-external-services)
13. [Privacy and security](#privacy-and-security)
14. [Known limitations](#known-limitations)
15. [Troubleshooting](#common-errors-and-fixes)
16. [Project structure](#project-structure)
17. [Licences and credits](#licences-and-credits)

---

## What it does

```
input ─► read content ─► extract claims ─► which are checkable? ─► search queries ─► web search (Serper)
  text      (URL fetch,      (AI, keeps        (AI: opinions are       (claim language    + read the
  URL        image OCR,       original          "Not checkable")        + English, with    full pages
  image      video frames)    language)                                 country/dates)
  audio ─► local transcription (faster-whisper) ─► you review/correct the transcript and pick statements ─► text
  video ─► FFmpeg extracts audio ─► same as audio;  frames read separately for on-screen text
                                                           ─► assess each evidence excerpt (AI, must quote it;
                                                              related videos/rumours and outdated reports are context only)
                                                           ─► fixed verdict rules ─► verdict first, then explanation, then linked sources
```

Verdicts: **True** · **False** · **Partly true / disputed** · **Unverified** (plus **Not checkable** for opinions).
Internally (and in the JSON download) these are `SUPPORTED`, `CONTRADICTED`, `MIXED`, `INSUFFICIENT_EVIDENCE` and `NOT_CHECKABLE`.

## Feature status

"Tested with mocks" means automated tests ran with simulated AI and search responses. "Live-tested" means it ran
against the real service on the development machine. **No AI or search API keys were available during development,
so the full end-to-end pipeline has not yet been run live.** See [what remains to test](#what-still-needs-a-live-test).

| Feature | Status |
|---|---|
| Text input → claims → evidence → verdicts → report | Implemented; tested end to end with mocks. **Live run pending API keys.** |
| Four verdict categories with documented rules | Implemented; rule logic fully unit-tested (no AI involved) |
| Source title, publisher, URL, published date, retrieval date, source type | Implemented; tested with mocked Serper responses |
| Country-biased search (Kenya `ke`, Uganda `ug`, Rwanda `rw`, ...) | Implemented; tested with mocks |
| Web link input (safe fetch + article extraction) | Implemented; **live-tested** on real pages, plus SSRF tests |
| Image input (PNG/JPEG/WebP; AI reads the text) | Implemented; image handling live-tested locally, AI step mocked |
| Audio input (MP3/WAV/M4A/OGG/Opus/FLAC/AAC/WebM; local faster-whisper) | Implemented; **live-tested** locally with the real `base` model on `demo_data/speech.mp3` |
| Video input (MP4/M4V/MOV/WebM; FFmpeg audio track + 8 sampled frames) | Implemented; FFmpeg audio extraction and real transcription **live-tested** locally on a short clip with speech; vision AI mocked |
| Transcript review (edit, pick timestamped statements, then fact-check) | Implemented; tested with mocks |
| Verdict-first results page; related-media and outdated reports kept as context | Implemented; rule logic unit-tested; **live run pending API keys** |
| English | Prompts designed for it; mocked tests |
| Swahili | Prompts keep the original language; mocked routing tests; **quality not yet measured live** |
| Luganda | **Experimental.** AI models and search engines have limited Luganda coverage. Mocked routing tests only |
| OpenAI provider | Implemented; tested with mocks |
| Anthropic Claude provider | Repaired (current model, JSON handling, usage, refusals); tested with mocks |

## Requirements

- Windows 10/11 (macOS/Linux also work), **Python 3.11 or 3.12**
- Internet access
- API keys: **Serper** (search) and **OpenAI**, **Anthropic** or **Gemini** (AI). Speech transcription needs **no** key.
- **FFmpeg** to extract audio from videos (also used to decode audio files when present). If it is not on your PATH,
  set `FFMPEG_PATH` in `.env`. (On the development machine ffmpeg was found in the GNU Octave folder.)
- About 1 GB free RAM and ~150 MB disk for the default `base` speech model.

## Windows setup (step by step)

Open **PowerShell** in VS Code (Terminal → New Terminal) and go to the project folder:

```powershell
cd "C:\Users\Admin\Downloads\OpenFactVerification-main-1\OpenFactVerification-main\OpenFactVerification-main"
```

1. **Create a virtual environment** (a private folder of packages just for this project):
   ```powershell
   py -3.12 -m venv .venv
   ```
2. **Activate it** (your prompt will start with `(.venv)`):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell says scripts are disabled, run this once, then activate again:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
3. **Install the dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## API keys and configuration

1. Copy the template: `Copy-Item .env.example .env`
2. Open `.env` in VS Code and fill in:
   - `SERPER_API_KEY` from <https://serper.dev> (sign up, then API Key)
   - `OPENAI_API_KEY` from <https://platform.openai.com/api-keys>, **and/or** `ANTHROPIC_API_KEY` from <https://console.anthropic.com>
3. Optional: choose the provider/model with `TRUTHLENS_PROVIDER` (`openai` or `claude`) and `TRUTHLENS_MODEL`.
   Defaults are `gpt-4.1-mini` (OpenAI) and `claude-opus-5-5` (Claude). Model availability changes over time;
   use a model ID that your account can access. Cheaper Claude choices are `claude-sonnet-5-5` and `claude-haiku-5-5`.

**Configuration precedence** (highest first): a non-empty value in an optional YAML file passed with
`--api_config`, then real environment variables, then `.env`. `.env` is listed in `.gitignore`, so it is never committed.
Keys are never printed, logged, or sent to the browser. Do not paste keys into chat or screenshots.

## Audio and video transcription (local, free)

Speech is transcribed with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), an open-source
implementation of OpenAI's Whisper model that runs **on your own computer**. There is no API key and no per-minute cost.

- **Install:** `pip install -r requirements.txt` installs `faster-whisper`. (`av<19` is pinned because faster-whisper 1.2
  is not compatible with PyAV 19.)
- **Model download:** the first transcription downloads the model once from Hugging Face (about 145 MB for `base`)
  and caches it in your user folder. This needs internet access. After that, audio never leaves your machine.
  To download it ahead of time: `python -m factcheck.utils.transcription --download`
- **Default configuration:** model `base`, `device=cpu`, `compute_type=int8`, beam size 5, voice-activity filter on
  (skips silence), first 10 minutes only. Change these in `.env` (`TRUTHLENS_WHISPER_MODEL=tiny|base|small|medium`, ...).
  `tiny` is fastest but least accurate; `small` is noticeably better for Swahili but about 3x slower than `base`.
- **Speed:** depends on your processor, the model and the length of the recording. On the development laptop
  (Intel i5-1135G7, 8 threads), `base` transcribed an 8-second clip in about 8 seconds, including warm-up
  (the first-ever run also spent about 19 seconds downloading and loading the model).
- **Languages:** English and Swahili are supported by Whisper. **Whisper has no Luganda model**, so Luganda speech is
  transcribed with automatic language detection and will contain many errors.
- **Try it on a file:** `python -m factcheck.utils.transcription demo_data/speech.mp3`

**FFmpeg.** Check it with `ffmpeg -version`. If it is missing on Windows: `winget install Gyan.FFmpeg`
(or download a build from <https://www.gyan.dev/ffmpeg/builds/>, unzip it, and set
`FFMPEG_PATH=C:\path\to\ffmpeg.exe` in `.env`), then open a new terminal.
macOS: `brew install ffmpeg`. Debian/Ubuntu: `sudo apt install ffmpeg`.

**How it works in the app:** choose *Audio* or *Video* and upload. The file is transcribed (videos: FFmpeg extracts the
audio track first; a few frames are read separately for on-screen text). You then see a **review page** with the
transcript and its timestamped statements: correct mistakes, tick the statements you want checked (or keep the whole
text), and press *Fact-check this text*. Nothing is sent to the AI model or search engine before you do this.

**Render (hosted) deployment.** Render's free web services have 512 MB RAM and 0.1 CPU and no persistent disk, so the
model would be re-downloaded after every restart, transcription would be very slow, and loading it risks running out of
memory and restarting the whole site. Therefore, when the app detects Render (the `RENDER` environment variable) and
`TRUTHLENS_TRANSCRIPTION` is not set, **transcription is switched off** and the form explains why; text, link and image
checks are unaffected. To enable it on a larger paid instance, set `TRUTHLENS_TRANSCRIPTION=on` (and consider
`TRUTHLENS_WHISPER_MODEL=tiny`). No paid transcription service is used anywhere.

## Running the web app

```powershell
.\.venv\Scripts\Activate.ps1
python webapp.py
```

You should see:

```
Kweli Lens AI | provider=gpt model=gpt-4.1-mini | LLM key: OK | Serper key: OK
Speech transcription: faster-whisper base | FFmpeg: found
Open http://127.0.0.1:2024 in your browser.
```

Open **http://127.0.0.1:2024**. The home page's *System status* box shows which services are configured.
Stop the server with `Ctrl+C`. Options: `python webapp.py --client claude --model claude-sonnet-5-5 --port 5000`.

## Example inputs

| Type | Try this | What a sensible result looks like |
|---|---|---|
| Text | `Kampala is the capital of Uganda. Mount Kilimanjaro is in Kenya.` (country: Uganda) | Claim 1 True; claim 2 False (Kilimanjaro is in Tanzania) |
| Text (numbers) | `Kenya introduced a new education policy in 2026, reducing university fees by 50%.` (country: Kenya) | Two claims; likely Unverified or Partly true unless sources state the 50% figure |
| Current event | `President Ruto is dead.` (country: Kenya) | Decided only from reports from the last 30 days; debunks of fake videos are listed as context, not as proof either way |
| Swahili | `Nairobi ni mji mkuu wa Kenya.` (language: Swahili, explain in Swahili) | True, explanation in Swahili |
| Luganda | `Kampala kye kibuga ekikulu ekya Uganda.` (language: Luganda) | Ideally True; results may be weaker (experimental) |
| Opinion | `This is the best government Uganda has ever had.` | Not checkable |
| URL | A public news article from monitor.co.ug, newvision.co.ug or nation.africa | Article title, publisher and date shown; claims checked |
| Image | `demo_data/image.webp` | Text read from the image is shown, then checked |
| Audio | `demo_data/speech.mp3` | Review page shows "The longest river in Africa is the Nile River ..." with timestamps; then check it |
| Video | `demo_data/video.m4v` | On-screen text from 8 frames; the app notes that this video has **no audio track** |

## Command line use

```powershell
python -m factcheck --modal string --input "Kampala is the capital of Uganda." --country Uganda
python -m factcheck --modal url --input "https://en.wikipedia.org/wiki/Kampala"
python -m factcheck --modal image --input demo_data/image.webp
python -m factcheck --modal audio --input demo_data/speech.mp3     # transcribes locally, checks the whole transcript
python -m factcheck.utils.transcription demo_data/speech.mp3       # transcript only, with timestamps
python -m factcheck --modal string --input "Nairobi ni mji mkuu wa Kenya." --language sw --output_language sw
```

Results are printed as JSON. As a Python library:

```python
from factcheck import FactCheck
fc = FactCheck(default_model="gpt-4.1-mini")
report = fc.check_text("Kampala is the capital of Uganda.", country="Uganda", output_language="en")
```

## Running the tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -v
```

Expected: **182 passed**. These tests are **offline and free**. AI models, search results and web pages are
simulated ("mocked"), so they check Kweli Lens's own logic (parsing, verdict rules, security, uploads, report rendering),
not the quality of the real AI. Images, videos and ffmpeg run for real; ffmpeg tests are skipped if ffmpeg is not installed.
One test (`test_real_whisper_transcribes_demo_speech`, marked `slow`) runs the real Whisper model on `demo_data/speech.mp3`;
it is skipped until the model has been downloaded. Run `python -m pytest -m "not slow"` to leave it out.

| Test file | Covers |
|---|---|
| `tests/test_parsing.py` | safe JSON parsing (no `eval`), invalid AI responses, claim extraction/checkworthiness/query fallbacks, invented quotes |
| `tests/test_verdict.py` | the four verdicts, independence, weights, no-evidence and conflicting-evidence cases, related-media and outdated-report rules, ranking |
| `tests/test_safe_fetch.py` | URL validation, SSRF/private network blocking, redirects, size and type limits, article extraction |
| `tests/test_retriever_and_clients.py` | Serper metadata, country bias, provider failures and timeouts, missing keys, OpenAI/Claude clients |
| `tests/test_multimodal.py` | image text extraction, local transcription (mocked + one real run), silent/corrupt/unsupported audio, model download failure, Render switch, video audio extraction, file type checks |
| `tests/test_pipeline.py` | full pipeline (simulated), insufficient/mixed evidence, the "President Ruto is dead" scenario, multilingual routing for 10 en/sw/lg claims |
| `tests/test_webapp.py` | CSRF, missing keys, XSS escaping, unsafe links, uploads, transcript review flow, verdict-first layout, private reports, error states |

### What still needs a live test

Once keys are in `.env`:

1. **Text end to end:** run the app and submit the text examples above. Open every source link and check that the
   excerpt really appears on the page and supports the stance shown.
2. **Multilingual evaluation** (costs a little money, see below):
   ```powershell
   python script/eval_multilingual.py
   ```
   It prints verdict accuracy and whether claims stayed in their original wording, per language. Have a fluent Swahili
   and Luganda speaker review `tests/data/multilingual_claims.json` (not yet done) and the explanations.
3. **Media:** upload `demo_data/image.webp`, `demo_data/speech.mp3` and a short news clip with speech, review the transcript
   and fact-check one statement.
4. **Current events:** check `President Ruto is dead.` (country Kenya) and confirm the verdict is based on recent, dated reports.

## How verdicts work

The AI only labels individual evidence excerpts. The verdict is computed by fixed rules in
[`factcheck/core/Verdict.py`](factcheck/core/Verdict.py) and explained in the app at `/about`:

- Only evidence that **supports**, **contradicts** or **partly supports** counts. Context and irrelevant results never move a verdict.
- Weight = basis (explicit quote verified in the source 1.0, inferred 0.5) × source type (government/academic/fact-checker 1.0,
  news 0.8, other 0.5, social media 0.25) × (0.7 if only the search snippet could be read).
- Several pages from the same website count as **one** source.
- A side is *established* at total weight ≥ 1.0. Both sides established, or the stronger side less than twice the weaker → **Mixed**.
  Only one side established → **Supported** / **Contradicted**. Neither → **Insufficient evidence**.
- **Unverified (insufficient evidence) never means false.** "Evidence strength" (low/moderate/high) is the weighted amount of evidence,
  **not** a probability that the claim is true.
- **Evidence must be about the claim itself.** The assessor marks evidence that is about a related video, post, rumour or a
  different event as `scope: related`; such evidence is shown under *Related reports and context* and never counted.
  A report that a video is fake does not show whether the claim in the video is true.
- **Current situations need recent evidence.** Claims such as "X is dead / has been arrested / resigned" with no date are
  time-sensitive: generated searches are limited to the past month, and deciding evidence older than 30 days is shown as
  context (an older report that someone died still supports a claim that they are dead).
- The results page shows the **verdict first**, then a 2-4 sentence plain-English explanation, then up to five key
  sources (ranked: deciding evidence, strongest and most recent first), then related reports, then the method details.
- Source types are guessed from the website address only. Being on a list does not make a source reliable.

## API costs and external services

| Service | Used for | Approximate cost (check current pricing) |
|---|---|---|
| Serper.dev | Google search results | 2,500 free queries on sign-up, then about US$1 per 1,000 queries. A typical check uses 2-4 queries per claim. |
| OpenAI (default `gpt-4.1-mini`) | claims, queries, evidence assessment, image reading | Roughly US$0.005-0.02 per claim (about 10 short AI calls per claim) |
| faster-whisper (local) | audio/video speech-to-text | Free; runs on your computer (one-time ~145 MB model download) |
| Anthropic Claude (optional) | same as OpenAI | `claude-opus-5-5` is about US$4 / US$20 per million input/output tokens: very roughly US$0.10-0.30 per claim. `claude-sonnet-5-5` and `claude-haiku-5-5` are cheaper. |

Each evidence excerpt is one AI call, so a long article with many claims costs more. Claude requests enable Anthropic's
server-side refusal fallback by default (`CLAUDE_REFUSAL_FALLBACK=1`); set it to `0` to turn it off.

## Privacy and security

What is in place:
- **No `eval()`** on AI output: responses are parsed as data and validated; malformed answers are discarded, not executed.
- **SSRF protection:** only http/https on ports 80/443; every hostname and redirect hop must resolve to public IPs (no
  localhost, private ranges, or cloud metadata addresses); timeouts and size limits; HTML/text only.
- **Uploads:** extension *and* file-content checks, size limits (image 10 MB, audio 50 MB, video 100 MB), random
  filenames in a temporary folder that is deleted after processing (recordings and extracted audio are deleted as soon as
  transcription ends). Uploaded files are never executed. FFmpeg is called with an argument list, never through a shell.
- **Transcription is local:** audio is not sent to any transcription service. Transcripts are kept only in memory for the
  review step (same 1-hour limit and session rule as reports).
- **Web app:** CSRF tokens, Content-Security-Policy, escaped output (XSS), only `http(s)` links clickable, `noreferrer`
  on source links, debug mode off, binds to `127.0.0.1` by default.
- **Data retention:** submissions and reports are kept **only in server memory**, for 1 hour (`TRUTHLENS_RESULT_TTL`),
  visible only to the browser session that created them, and are lost when the server stops. Use *Download JSON* to keep a report.
- **Secrets:** read from `.env`/environment, never logged or shown.

Remaining risks (not fully solved):
- Content you submit is sent to the AI provider (OpenAI/Anthropic) and search queries go to Serper/Google. Do not submit
  confidential sources or personal data you are not allowed to share with these services.
- DNS rebinding (a hostname changing its IP between our check and the connection) is not fully prevented.
- The built-in Flask server is for local use. For a shared/online deployment use a production server (e.g. `waitress`)
  behind HTTPS, set a fixed `FLASK_SECRET_KEY`, and add authentication and rate limiting.
- Prompt injection: a web page could contain text that tries to mislead the AI. Quote verification and fixed verdict
  rules reduce, but do not remove, this risk. Always check the sources.
- Server log files in `log/` record claims and queries (not keys). Delete them if the content is sensitive.

## Known limitations

- Search coverage of East African, local-language and offline sources is limited; absence of evidence is common.
- **Luganda is experimental** and Swahili quality is unmeasured; machine-written explanations may contain language errors.
- Pages behind paywalls/logins or built with heavy JavaScript cannot be read; the search snippet is used and flagged.
- Video: 8 frames from the first 10 minutes; brief on-screen text can be missed. Spoken content and on-screen text are
  extracted separately and may not agree.
- Text read from images and audio transcripts may contain errors. They show what the media says, not whether it is true.
- Statistical claims: Kweli Lens asks the AI to check that population, period and measure match, but it does not do its
  own data analysis. Look for the original dataset.
- The original Loki "google" scraping retriever needs extra heavy packages and is not maintained here; use `serper`.
- `templates/input.html` and `templates/LibrAI_fc.html` are the original Loki pages and are no longer used.

## Common errors and fixes

| Message | Fix |
|---|---|
| `Setup needed ... OPENAI_API_KEY is not set` | Create `.env` from `.env.example`, add the key, restart `python webapp.py` |
| `The evidence search service rejected the API key` | Check `SERPER_API_KEY` in `.env` (no quotes or spaces) |
| `rate limit or credit quota was reached` | Serper credits used up; top up or wait |
| `AI provider rejected the request (NotFoundError)` | The model name isn't available to your account; change `TRUTHLENS_MODEL` |
| `The speech-to-text model could not be loaded` | First use needs internet to download the model; check the connection and disk space |
| `Audio/video transcription is turned off on this hosted server` | Expected on Render free instances; run locally, or set `TRUTHLENS_TRANSCRIPTION=on` on a larger instance |
| `needs the faster-whisper package` | `pip install -r requirements.txt` inside the activated `.venv` |
| `No speech was detected in the recording` | The file is silent, music-only or very quiet |
| `ffmpeg not found` / `FFmpeg is not installed` | Install FFmpeg (see above) or set `FFMPEG_PATH=C:\path\to\ffmpeg.exe` in `.env` |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError` | Activate `.venv` first, then `pip install -r requirements.txt` |
| `Readable article text could not be extracted` | The site blocks tools or needs login/JavaScript; paste the article text instead |
| Report link says "not found" | Reports expire after 1 hour or when the server restarts, and only open in the browser that created them |

## Project structure

```
webapp.py                     Flask web app (forms, background jobs, reports, security headers)
factcheck/__init__.py         FactCheck pipeline orchestrator (check_text)
factcheck/__main__.py         command-line interface
factcheck/inputs.py           turns text/URL/image/audio/video into text; upload limits
factcheck/core/Decompose.py   claim extraction            core/CheckWorthy.py   verifiable or not
factcheck/core/QueryGenerator.py  search queries          core/ClaimVerify.py   per-evidence assessment
factcheck/core/Verdict.py     fixed verdict rules         core/Retriever/serper_retriever.py  web search + metadata
factcheck/utils/llmclient/    OpenAI, Claude and local-model clients
factcheck/utils/prompt/truthlens_prompt.py  the prompts (multilingual, grounded)
factcheck/utils/safe_fetch.py SSRF-safe fetching, article text extraction
factcheck/utils/sources.py    domain, source type, publication date metadata
factcheck/utils/multimodal.py image / audio / video processing
factcheck/utils/transcription.py  local speech-to-text (faster-whisper), model loading and settings
factcheck/utils/json_utils.py safe parsing of AI output
templates/, assets/css/truthlens.css, assets/js/truthlens.js   user interface
tests/                        automated tests (offline);  tests/data/multilingual_claims.json
script/eval_multilingual.py   live multilingual evaluation (needs keys)
```

## Licences and credits

- Kweli Lens AI is a modification of **Loki / OpenFactVerification**, © 2024 LibrAI, released under the
  [MIT License](LICENSE). The original copyright and licence notice are preserved in `LICENSE`.
- Python dependencies: anthropic (MIT), openai (Apache-2.0), Flask (BSD-3-Clause), beautifulsoup4 (MIT),
  opencv-python-headless (Apache-2.0; bundles FFmpeg libraries under LGPL), requests (Apache-2.0), httpx (BSD-3-Clause),
  python-dotenv (BSD-3-Clause), PyYAML (MIT), tiktoken (MIT), backoff (MIT), pytest (MIT),
  faster-whisper (MIT), CTranslate2 (MIT), PyAV (BSD-3-Clause). Whisper model weights (MIT, by OpenAI, converted by
  SYSTRAN) are downloaded at first use, not bundled.
- ffmpeg is an external program (LGPL/GPL) and is not distributed with this project.
- External services (OpenAI, Anthropic, Serper/Google) have their own terms of use and data policies; review them
  before processing sensitive material.
- No datasets or model weights are bundled. The demo files in `demo_data/` come from the original Loki repository.

### Cite Loki
```
@misc{li2024lokiopensourcetoolfact,
      title={Loki: An Open-Source Tool for Fact Verification},
      author={Haonan Li and Xudong Han and Hao Wang and Yuxia Wang and Minghan Wang and Rui Xing and Yilin Geng and Zenan Zhai and Preslav Nakov and Timothy Baldwin},
      year={2024},
      eprint={2410.01794},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.01794},
}
```
