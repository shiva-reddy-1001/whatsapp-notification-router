# Backlog

Prioritized work that is intentionally not represented as completed behavior.

## Reliability

- Add provider-specific retry classification: retry transient connection,
  HTTP 429, and 5xx failures; do not retry authentication or invalid-model
  errors.
- Add exponential backoff with jitter and a total per-message retry budget.
- Split connection, read, and whole-request timeouts; enforce a run-level
  deadline and record timeout telemetry without message contents or secrets.
- Persist failed message IDs and error categories so interrupted runs can resume
  without reprocessing successful classifications.
- Write predictions incrementally to a validated checkpoint file and atomically
  publish `output.csv` only after all rows pass validation.

## Multimodal history

- Extract and cache OCR/transcripts for historical image and voice messages,
  then include their normalized content in retrieval.
- Add a Qwen vision adapter for images where OCR is empty or low quality;
  extract document type, dates, amounts, URLs, QR/payment cues, and urgency.
- Record ASR language, duration, and confidence rather than a fixed quality
  score; audit numeric warnings from local Whisper inference.

## Retrieval and embeddings

- Build candidate sets by recipient plus sender/group/business before ranking.
- Add recency, interaction outcome, message type, risk-pattern, and campaign
  duplicate features to the retrieval score.
- Benchmark lexical retrieval against a small local embedding model. Store
  normalized embeddings in SQLite first; add a vector index only when corpus
  size or measured latency justifies it.
- Emit an evidence ID only when its interaction outcome or similarity directly
  contributes to the final reason.

## Evaluation and submission

- Add regression thresholds for action/type accuracy by conversation and media
  type, plus evidence precision and confidence calibration.
- Add clean-run tests for OpenAI and Ollama provider modes using recorded,
  secret-free response fixtures.
- Add a submission manifest with runtime versions, prompt version, model name,
  output checksum, and evaluation summary.
