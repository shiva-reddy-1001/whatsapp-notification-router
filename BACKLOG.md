# Backlog

Prioritized work that is intentionally not represented as completed behavior.

## Reliability

- Split connection and read timeouts (whole-request timeout, per-message retry
  deadline, and optional run deadline are implemented).
- Persist failed message IDs and error categories so interrupted runs can resume
  without reprocessing successful classifications.
- Optionally write predictions to a validated checkpoint for cross-process
  resume. Final `output.csv` publication is already atomic.

## Multimodal history

- Record ASR language, duration, and confidence rather than a fixed quality
  score; audit numeric warnings from local Whisper inference.

## Retrieval and embeddings

- Build candidate sets by recipient plus sender/group/business before ranking.
- Add campaign-duplicate features to the retrieval score; recency and
  risk-pattern agreement are implemented.
- Benchmark and calibrate hybrid-score weights against labeled evidence; add a
  vector index only when corpus size or measured latency justifies it.
- Emit an evidence ID only when its interaction outcome or similarity directly
  contributes to the final reason.

## Evaluation and submission

- Validate the calibrated type boundaries with a larger, independent set:
  urgent/event, business-update/promotion, spam/personal, safety-advisory/scam,
  and unknown/personal. Do not add sample-ID or phrase-specific corrections.
- Add automated regression thresholds by media type, plus evidence precision
  and confidence calibration. Conversation/type slices and 30 solved examples
  currently pass at `1.000` for action and type locally.
- Add clean-run tests for OpenAI and Ollama provider modes using recorded,
  secret-free response fixtures.
