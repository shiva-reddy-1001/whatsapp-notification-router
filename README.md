# WhatsApp Message Notification Router

AI-powered, personalized routing for WhatsApp text, image, and voice messages.
For each row in `dataset/messages.csv`, the router decides whether to
`notify`, `digest`, or `mute` and writes a schema-validated `output.csv`.

This repository is the complete runnable solution for the HackerRank
Orchestrate Message Notification Router challenge. The fastest evaluator path
is [Quick start](#quick-start-openai--judge-run). The original contract is in
[`problem_statement.md`](./problem_statement.md).

## Submission contract

The program reads only participant-facing files under `dataset/` and writes
exactly one result for every incoming `message_id` with these columns, in this
exact order:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

- `action`: `notify`, `digest`, or `mute`
- `message_type`: `personal`, `urgent`, `event`, `payment`,
  `business_update`, `promotion`, `greeting`, `forward`, `spam`, `scam`, or
  `unknown`
- `confidence`: finite number from `0` through `1`
- `evidence_message_ids`: semicolon-separated historical IDs, or `none`

The writer rejects incomplete runs, duplicates, invalid labels, invalid
confidence, unknown evidence IDs, and wrong column order. It publishes the CSV
atomically only after all messages succeed.

## Prerequisites

| Requirement | OpenAI / judge | Local Ollama |
|---|---:|---:|
| Python 3.9+ (verified on 3.9.6) | required | required |
| Internet access | OpenAI API calls | first model/Whisper download only |
| `OPENAI_API_KEY` | required | no |
| Ollama | no | required |
| `qwen2.5vl:3b` | no | required (about 3.2 GB) |
| `nomic-embed-text` | no | required (about 274 MB) |
| FFmpeg | no | required for local voice notes |
| Tesseract OCR | optional | recommended; Qwen vision still analyzes images |

The Python dependencies are pinned in `requirements.txt`. A local run may
download the faster-whisper `tiny` model on its first voice note. Model weights,
caches, datasets, and secrets are deliberately excluded from `code.zip`.

## Quick start: OpenAI / judge run

Run from the extracted repository root. The judge can inject only
`OPENAI_API_KEY`; `ROUTER_LLM_PROVIDER=auto` is already the default and selects
OpenAI without source edits or local models.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# Export the real key securely in the runner environment; never commit it.
export OPENAI_API_KEY="<injected-by-runner>"

# This validates provider selection and required credentials without inference.
.venv/bin/python -m code.main --check-config --provider auto

# Full run. --input is intentionally restricted to this participant file.
.venv/bin/python -m code.main \
  --dataset-dir dataset \
  --input dataset/messages.csv \
  --output dataset/output.csv
```

With OpenAI selected, the same provider strategy supplies structured type and
action decisions, image understanding, voice transcription, and embeddings.
No Ollama server, local model, FFmpeg, or Tesseract is needed in this mode.

`OPENAI_MODEL` defaults to `gpt-4.1-mini`, the embedding model defaults to
`text-embedding-3-small`, and transcription defaults to `whisper-1`. Override
these only with compatible model IDs available to the supplied API key.

## Full local setup: Ollama

### 1. Install system tools

On macOS with Homebrew (include `pyenv` when Python 3.9 is not already
available):

```bash
brew install pyenv ollama ffmpeg tesseract
pyenv install -s 3.9.6
pyenv local 3.9.6
```

On Linux, install Ollama using its official installer and install `ffmpeg` and
`tesseract-ocr` with the operating system package manager. Start Ollama in a
separate terminal if it is not already running:

```bash
ollama serve
```

### 2. Pull the memory-conscious models

```bash
ollama pull qwen2.5vl:3b
ollama pull nomic-embed-text
ollama list
```

The selected generation model was chosen for a 16 GB development machine and
uses roughly 3–4 GB while active. Qwen handles both classification and semantic
image analysis; the small embedding model supports historical retrieval.

### 3. Create the Python environment

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The solution is tested with Python 3.9.6. If several Python versions are
installed, create the environment with that interpreter (for example,
`pyenv local 3.9.6` before `python -m venv .venv`). On Windows, use
`.venv\Scripts\python` instead of `.venv/bin/python` and set environment
variables using the active shell's syntax.

### 4. Validate and run

```bash
ROUTER_LLM_PROVIDER=ollama .venv/bin/python -m code.main --check-config

# Optional one-time historical media and embedding warm-up. A normal run also
# performs this automatically.
ROUTER_LLM_PROVIDER=ollama .venv/bin/python -m code.index_history \
  --dataset-dir dataset

ROUTER_LLM_PROVIDER=ollama .venv/bin/python -m code.main \
  --dataset-dir dataset \
  --input dataset/messages.csv \
  --output dataset/output.csv
```

For the final reproducibility rehearsal, discard only router-owned cached
media, embeddings, and predictions and rebuild them:

```bash
ROUTER_LLM_PROVIDER=ollama .venv/bin/python -m code.main \
  --dataset-dir dataset \
  --input dataset/messages.csv \
  --output dataset/output.csv \
  --clear-cache
```

## Environment configuration

No `.env` file is required. Environment variables override checked-in defaults;
explicit CLI flags override the corresponding path/provider variables. For
local convenience only:

```bash
cp .env.example .env
```

`.env` is ignored by git. Never put a real key in `.env.example`, source code,
logs, output, or the submission archive.

Important variables:

| Variable | Default | Purpose |
|---|---|---|
| `ROUTER_LLM_PROVIDER` | `auto` | `auto`, `openai`, or `ollama` |
| `OPENAI_API_KEY` | unset | canonical judge/API credential |
| `OPENAI_MODEL` | `gpt-4.1-mini` | OpenAI classification/vision model |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | local Ollama endpoint |
| `OLLAMA_MODEL` | `qwen2.5vl:3b` | local classifier and vision model |
| `ROUTER_MEDIA_MODE` | `auto` | `auto` or `off` |
| `ROUTER_VISION_PROVIDER` | `auto` | `auto`, `none`, `openai`, or `ollama` |
| `ROUTER_AUDIO_PROVIDER` | `auto` | `auto`, `none`, `openai`, or `local` |
| `ROUTER_EMBEDDING_PROVIDER` | `auto` | `auto`, `none`, `openai`, or `ollama` |
| `ROUTER_CACHE_PATH` | `.router-cache/router.sqlite` | ignored resumable cache |
| `ROUTER_REQUEST_TIMEOUT_SECONDS` | `60` | timeout for one provider request |
| `ROUTER_MAX_RETRIES` | `2` | bounded transient retries |
| `ROUTER_RETRY_MODE` | `exponential` | `none`, `fixed`, or `exponential` |
| `ROUTER_MESSAGE_DEADLINE_SECONDS` | `180` | total budget for one message operation |
| `ROUTER_RUN_DEADLINE_SECONDS` | `0` | optional whole-run deadline; `0` disables |

All supported variables and safe defaults are documented in `.env.example`.
Preflight verifies OpenAI key presence and verifies the local Ollama service and
model. Actual OpenAI key validity/model access is confirmed by the first API
request. An explicitly selected provider never silently switches classification
policy. Retries are limited to transient connection/timeout/429/5xx and
repairable structured output failures; authentication and unknown-model errors
fail immediately when returned by the provider.

## How the solution works

```text
participant CSVs
  -> validated per-message case file (user + group/business + load context)
  -> current and historical media extraction (OCR/vision or ASR)
  -> auditable risk, priority, fatigue, and relationship features
  -> cached hybrid historical retrieval (dense + lexical + outcomes)
  -> evidence-isolated message-type specialist
  -> personalized action classifier
  -> narrow safety/consistency policy + calibrated confidence
  -> atomic output contract validation
```

Key design choices:

- **Multimodal content is first-class.** Current and historical images are
  canonicalized, OCR'd, and semantically analyzed. Voice notes are transcribed.
  Conflicting or poor-quality media lowers confidence.
- **Personalization is explicit.** The case file separates recipient behavior,
  group membership/mute state, business trust/opt-in history, daily load, and
  prior reactions rather than dumping raw CSV rows into one prompt.
- **Retrieval is attributable.** Historical text and extracted media are
  embedded once, cached in SQLite, and combined with lexical, relationship, and
  user-outcome signals. Only same-user, retrieval-approved IDs can be emitted.
- **Type and action are separated.** A history-isolated specialist determines
  semantic `message_type`; a second view uses personalized evidence for the
  routing action. This prevents retrieved vocabulary from contaminating type.
- **The model remains primary.** A narrow final policy enforces challenge
  invariants for unsafe credential/payment requests, explicit deferral,
  opted-out/rejected promotions, urgent interruptions, reason consistency, and
  confidence caps. There is no general rules-based classifier fallback.
- **Execution is resumable and deterministic where practical.** Versioned
  prompt/cache keys, temperature `0`, seed `42`, bounded retries, deadlines,
  preflight checks, and an atomic schema guard make failures visible and reruns
  safe.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for process diagrams,
[`.design.md`](./.design.md) for low-level design, and
[`DECISIONS.md`](./DECISIONS.md) for assumptions and calibration decisions.

- [Low-level design](./.design.md)
- [Architecture and process flows](./ARCHITECTURE.md)
- [Decision record](./DECISIONS.md)
- [Known backlog](./BACKLOG.md)
- [V9 verification manifest](./SUBMISSION.md)

## Verified results

The latest V9 cold-cache local rehearsal produced:

| Check | Result |
|---|---:|
| Incoming/final rows | 110 / 110 |
| Exact schema, unique IDs, allowed labels, valid evidence | pass |
| Current + historical media with non-empty extraction | 33 / 33 |
| Historical assets enriched | 23 / 23 |
| Unit/regression tests | 46 passed |
| Solved calibration action accuracy | 1.000 (30 / 30) |
| Solved calibration type accuracy | 1.000 (30 / 30) |
| Extracted-package compilation/tests | pass |
| OpenAI judge static configuration dry preflight | pass |
| Real OpenAI end-to-end inference | requires an injected valid key; not claimed |

The 30 solved examples are a calibration gate, not a hidden test set or a claim
of leaderboard accuracy. Earlier dense retrieval alone measured action `0.500`
and type `0.367`; separating type reasoning and tightening generic taxonomy and
safety boundaries closed the observed calibration errors without hardcoding
sample IDs. Hidden-set performance remains the actual submission measure.

The verified V9 `dataset/output.csv` SHA-256 is recorded in
[`SUBMISSION.md`](./SUBMISSION.md). Regenerating through a remote or stochastic
provider can legitimately change predictions and therefore the checksum.

## Test, evaluate, and package

```bash
# Fast deterministic tests (write bytecode outside the repository).
PYTHONPYCACHEPREFIX=/tmp/router-pycache \
  .venv/bin/python -m unittest discover -v

# Score against the 30 solved calibration examples.
ROUTER_LLM_PROVIDER=ollama .venv/bin/python -m code.evaluation.main \
  --dataset-dir dataset --provider ollama

# Show isolated type confusions when calibrating.
ROUTER_LLM_PROVIDER=ollama .venv/bin/python -m code.evaluation.main \
  --dataset-dir dataset --provider ollama --type-only --show-errors

# Build the required clean archive.
.venv/bin/python scripts/package_submission.py --destination code.zip

# Inspect before upload: no dataset, .env, cache, venv, weights, or bytecode.
unzip -l code.zip
shasum -a 256 code.zip dataset/output.csv
```

`code.zip` contains the runnable `code/`, prompts/configuration, pinned
requirements, tests, packaging script, README, design/architecture records, and
the problem statement. The dataset is intentionally not duplicated in the code
archive because the runner supplies it and `output.csv` is uploaded separately.

## Troubleshooting

- **`ProviderUnavailable` or connection refused:** run `ollama serve`, confirm
  `ollama list`, or supply a valid `OPENAI_API_KEY` and select `auto/openai`.
- **Model not found:** pull the exact Ollama names shown above; model aliases
  must match the environment configuration.
- **Voice-note failure locally:** confirm `ffmpeg -version`. The first
  faster-whisper run also needs network access to download `tiny`.
- **Poor/missing OCR:** confirm `tesseract --version`; Qwen/OpenAI vision still
  provides semantic analysis when configured.
- **Timeouts/429/5xx:** increase request/message deadlines or retry limits in
  the environment. The default retry policy is bounded and exponential.
- **Stale local results:** use `--clear-cache`. It clears only the configured
  router SQLite cache, never the dataset or output path.
- **Wrong input rejected:** `--input` must resolve exactly to
  `<dataset-dir>/messages.csv`; organizer-only files cannot be routed.
- **No output after failure:** expected. Partial predictions are never
  published; fix the reported provider/media error and rerun.

## Final submission checklist

The challenge requires exactly three uploads:

1. **`code.zip`** — generated by `scripts/package_submission.py`.
2. **`output.csv`** — upload the verified `dataset/output.csv` under this name.
3. **`chat_transcript`** — upload the external append-only development log from
   `$HOME/hackerrank_orchestrate_august26/log.txt` (or the corresponding
   `%USERPROFILE%` path on Windows), with secrets redacted.

Before submitting:

- confirm the extracted archive installs and `python -m unittest discover -v`
  passes;
- confirm the runner places all participant CSV/media files under `dataset/`;
- run the exact OpenAI or Ollama command above from the repository root;
- confirm `dataset/output.csv` has the exact six columns and one row per input;
- ensure `.env`, API keys, `.router-cache`, model weights, and organizer-only
  files are absent from `code.zip`;
- upload the final CSV separately (the packager intentionally excludes it);
- upload the chat transcript separately; `AGENTS.md` is operational guidance,
  not a substitute for the transcript file.

No additional Markdown file is required by the runner. `README.md` is the
runner entry point; the other checked-in Markdown files provide traceability and
do not need special invocation.
