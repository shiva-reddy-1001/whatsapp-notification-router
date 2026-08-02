# Decision log

This file records deliberately conservative choices that should be revisited
after sample-set evaluation rather than silently becoming permanent behavior.

| Decision | Current choice | Why / follow-up |
|---|---|---|
| Default provider | `auto`: OpenAI if key is set, otherwise Ollama | No rule-based fallback; provider preflight is mandatory. |
| Local generation/vision model | `qwen2.5vl:3b` | Tested at ~3 GB active GPU. Evaluate JSON reliability on samples. |
| Audio model | faster-whisper `tiny`, CPU `int8` | Small and verified; compare `base` only if transcription quality limits routing. |
| Image path | OCR + Qwen vision with canonical JPEG bytes | Handles mislabeled AVIF/WEBP and bounded large images; cached by file mtime/model/version. |
| Historical media | Pre-enrich all 19 images and 4 voice notes | Makes media content retrievable; validated 23/23 enrichment locally. |
| Retrieval | cached dense + lexical + relationship/outcome hybrid | `nomic-embed-text` locally or `text-embedding-3-small` with an injected OpenAI key; SQLite is sufficient here. |
| Reliability | typed transient retries with configurable backoff/deadlines | Do not retry auth/model errors; retain strict final output validation. |
| Safety | credential/payment patterns are facts; provider owns classification | No rules-based routing flow; inspect misses on legitimate and scam payment messages. |
| Ambiguity | conservative `digest` | Tune only with sample evidence; avoid unnecessary interrupts. |
| Dataset size | 110 parsed target rows / 30 parsed solved sample rows | Corrected after using a CSV parser; multiline text makes shell line counts misleading. |
| Dense ablation | action 0.500 vs lexical 0.467; type 0.367 in both | Dense retrieval gives a modest action gain on 30 samples; weights and prompt/type policy still need calibration. |
