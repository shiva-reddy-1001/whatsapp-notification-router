# Decision log

This file records deliberately conservative choices that should be revisited
after sample-set evaluation rather than silently becoming permanent behavior.

| Decision | Current choice | Why / follow-up |
|---|---|---|
| Default provider | `auto`: OpenAI if key is set, otherwise Ollama | Allows judge key injection; force `rules` in deterministic tests. |
| Local generation/vision model | `qwen2.5vl:3b` | Tested at ~3 GB active GPU. Evaluate JSON reliability on samples. |
| Audio model | faster-whisper `tiny`, CPU `int8` | Small and verified; compare `base` only if transcription quality limits routing. |
| Image path | OCR first | Vision understanding is not yet passed to Ollama; add after measuring OCR misses. |
| Retrieval | lexical + relationship score, max 3 evidence rows | Dataset is small; add embeddings only if sample retrieval is inadequate. |
| Safety | credential/payment scam patterns override model | Inspect false positives on legitimate banking/delivery updates. |
| Ambiguity | conservative `digest` | Tune only with sample evidence; avoid unnecessary interrupts. |
| Dataset size | 110 parsed target rows / 30 parsed solved sample rows | Corrected after using a CSV parser; multiline text makes shell line counts misleading. |
