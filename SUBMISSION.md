# Submission Verification Manifest

Generated for the V9.1 submission-ready documentation checkpoint on 2026-08-02.

## Runtime contract

- Python: 3.9.6
- Local classifier and vision: `qwen2.5vl:3b` (`fb90415cde1e`, 3.2 GB)
- Local embeddings: `nomic-embed-text:latest` (`0a109f422b47`, 274 MB)
- Local audio: faster-whisper `tiny`, CPU `int8`; FFmpeg 8.1.2
- Judge mode: `OPENAI_API_KEY` selects OpenAI classification, vision,
  transcription, and embeddings through `ROUTER_LLM_PROVIDER=auto`
- Action prompt: `router-action-v10-safety-evidence-media`
- Type prompt: `router-type-v4-native-media-boundaries`
- Decision policy: `decision-policy-v7-trusted-deadlines-and-reason-consistency`

## Verification results

- Cold-cache end-to-end run: passed, 110/110 rows written
- Historical media enrichment: 23/23 historical assets
- Final media cache: 33/33 assets non-empty, zero zero-quality entries
- Unit/regression tests: 46 passed
- Solved calibration set: action `1.000`, type `1.000` on 30 rows
- Output contract: exact columns/order, 110 unique IDs, allowed labels,
  finite confidence, non-empty reasons, valid same-user evidence
- OpenAI judge static configuration dry preflight: passed using a non-secret
  placeholder key; no real-key end-to-end API result is claimed
- Extracted-package compilation and tests: passed
- README clean-machine, environment, judge-runner, troubleshooting, packaging,
  results-scope, and three-artifact upload audit: passed

Final `dataset/output.csv` SHA-256:

```text
4042afa158bbe4b04bda61cf5022256623417a07139bef01f8425eb2bc88db11
```

The solved sample is a calibration gate, not hidden-set ground truth. The final
CSV was also independently reviewed for credential/payment safety, explicit
deferral, promotion consent, historical-evidence direction, multimodal
conflicts, reason/action consistency, and confidence diversity.
