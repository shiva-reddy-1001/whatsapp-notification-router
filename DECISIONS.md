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
| Safety | explicit contract invariant after provider reasoning | Credential, sensitive-financial-detail, domain-mismatch, and pressured-advance-payment facts force `mute/scam`; legitimate “never share OTP” advisories are exempt. |
| Ambiguity | conservative `digest` | Tune only with sample evidence; avoid unnecessary interrupts. |
| Dataset size | 110 parsed target rows / 30 parsed solved sample rows | Corrected after using a CSV parser; multiline text makes shell line counts misleading. |
| Dense ablation | action 0.500 vs lexical 0.467; type 0.367 in both | Dense retrieval gives a modest action gain on 30 samples; weights and prompt/type policy still need calibration. |
| Type calibration | evidence-isolated specialist plus narrow taxonomy-boundary policy | Current solved calibration is `1.000` type and `1.000` action on 30/30; invariants use generic content/source facts, never sample IDs. Hidden-set auditing remains required. |
| Action calibration | safety → consent/noise → deferral → immediate operations → provider ambiguity | Prevents urgency language from defeating safety and prevents ads/chain forwards from notifying by default. |
| Confidence | weighted action/type/evidence alignment with media penalties and override caps | Avoids V8’s near-constant 0.90 and lowers certainty for missing or conflicting media. |
| Judge multimodal mode | OpenAI for classify/vision/audio/embed when key is injected | Normal `auto` command is end-to-end without Ollama, Whisper downloads, or source changes. |
| Output publication | validate then atomic same-directory replace | A partial/crashed run cannot leave a half-written submission CSV. |
