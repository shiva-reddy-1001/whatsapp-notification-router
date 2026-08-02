# Backlog

Prioritized work that is intentionally not represented as completed behavior.

## Reliability

- Split connection and read timeouts (whole-request timeout, per-message retry
  deadline, and optional run deadline are implemented).
- Persist failed message IDs and error categories so interrupted runs can resume
  without reprocessing successful classifications.
- Write predictions incrementally to a validated checkpoint file and atomically
  publish `output.csv` only after all rows pass validation.

## Multimodal history

- Record ASR language, duration, and confidence rather than a fixed quality
  score; audit numeric warnings from local Whisper inference.

## Retrieval and embeddings

- Build candidate sets by recipient plus sender/group/business before ranking.
- Add recency, message type, risk-pattern, and campaign
  duplicate features to the retrieval score.
- Benchmark and calibrate hybrid-score weights against labeled evidence; add a
  vector index only when corpus size or measured latency justifies it.
- Emit an evidence ID only when its interaction outcome or similarity directly
  contributes to the final reason.

## Evaluation and submission

- Add regression thresholds for action/type accuracy by conversation and media
  type, plus evidence precision and confidence calibration.
- Add clean-run tests for OpenAI and Ollama provider modes using recorded,
  secret-free response fixtures.
- Add a submission manifest with runtime versions, prompt version, model name,
  output checksum, and evaluation summary.
