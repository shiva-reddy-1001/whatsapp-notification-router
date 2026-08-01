# Architecture Guide

This document describes the executable V2 router architecture. It complements
[.design.md](./.design.md), which records the design rationale, and
[DECISIONS.md](./DECISIONS.md), which tracks decisions still open to tuning.

## System purpose

The router assigns every incoming WhatsApp message one action:

- `notify` — interrupt now;
- `digest` — show later; or
- `mute` — suppress.

It produces a schema-validated `dataset/output.csv` and preserves the history
used as evidence for each decision.

## End-to-end flow

```mermaid
flowchart LR
    A["Participant CSVs and media"] --> B["Dataset loader"]
    B --> C["Normalized Message and CaseFile"]
    C --> D["Media processor: OCR / ASR"]
    D <--> E[("SQLite cache")]
    D --> F["Feature extractor"]
    F --> G["History retrieval"]
    G --> H["Feature and policy facts"]
    H --> I["Structured provider classifier"]
    I <--> E
    I --> J
    J --> K["Output contract validator"]
    K --> L["dataset/output.csv"]
```

The provider consumes explicit safety, urgency, relationship, and noise facts
in a versioned case file. Schema validation is mandatory; an unavailable
provider fails before processing rather than silently changing policy.

## Component ownership

| Component | Responsibility | Durable output |
|---|---|---|
| `data_loader.py` | Read CSVs, normalize nulls/types, build indexes, create case files | In-memory indexes |
| `media_processor.py` | OCR image text and transcribe voice notes | Cached text and quality |
| `features.py` | Produce risk, priority, and noise/fatigue facts | Auditable case facts |
| `retrieval.py` | Select relevant prior messages and interaction outcomes | Evidence IDs and rationale |
| `prompting.py` | Define the versioned case-file prompt contract | `router-casefile-v3` |
| `providers.py` | Call OpenAI or Ollama and validate structured JSON | Cached predictions |
| `output_writer.py` | Enforce the evaluator’s output contract | `output.csv` |

## Case-file assembly

```mermaid
flowchart TD
    M["Incoming message"] --> T["Native text"]
    M --> X["Media reference"]
    X --> O["Image OCR"]
    X --> V["Voice transcription"]
    T --> C["Normalized content"]
    O --> C
    V --> C
    U["User profile and notification load"] --> CF["CaseFile"]
    G["Group and membership context"] --> CF
    B["Business and relationship context"] --> CF
    C --> CF
    CF --> F["Risk / priority / noise facts"]
    CF --> R["Relevant historical evidence"]
    F --> D["Decision stage"]
    R --> D
```

The `CaseFile` is the only object passed to a classifier. This keeps CSV join
logic, media extraction, and model prompting independent and testable.

## Classification policy

```mermaid
flowchart TD
    S["CaseFile with features and evidence"] --> P["Provider classification"]
    P --> V{"Allowed labels, evidence, confidence?"}
    V -->|"yes"| O["Prediction"]
    V -->|"no"| E["Retry once, then fail run"]
```

The prompt supplies safety facts but the provider owns the action decision. A
model result is rejected if its action/type is outside the allowed vocabulary,
its confidence is outside `[0, 1]`, or it cites an ID retrieval did not supply.

## Provider and fallback modes

```mermaid
flowchart LR
    C["Configuration"] --> P{"ROUTER_LLM_PROVIDER"}
    P -->|"openai"| O["OpenAI adapter"]
    P -->|"ollama"| L["Ollama adapter"]
    P -->|"auto with OPENAI_API_KEY"| O
    P -->|"auto without key"| L
    O --> V["JSON validation"]
    L --> V
    V -->|"valid"| OUT["Prediction"]
    V -->|"invalid"| ERR["Retry once, then fail"]
```

`auto` supports judge execution when `OPENAI_API_KEY` is injected, while local
development can force Ollama without a cloud credential. Provider preflight is
required before a run.

## SQLite cache and resumability

```mermaid
sequenceDiagram
    participant Run as Router run
    participant Cache as SQLite cache
    participant Media as Media processor
    participant Model as Provider

    Run->>Cache: Lookup media key (path, mtime, extractor config)
    alt Media hit
        Cache-->>Run: Cached OCR or ASR result
    else Media miss
        Run->>Media: Extract text
        Media-->>Cache: Store text and quality
    end
    Run->>Cache: Lookup prediction key (provider, model, prompt version, case file)
    alt Prediction hit
        Cache-->>Run: Cached validated prediction
    else Prediction miss
        Run->>Model: Structured case-file prompt
        Model-->>Run: JSON result
        Run->>Cache: Store validated result
    end
```

Cache keys include the policy/model inputs, so a media update, model change, or
prompt version change automatically causes a safe cache miss. The cache is
ignored by Git and contains no credentials.

## Output contract and quality gates

```mermaid
flowchart LR
    P["Predictions"] --> V1{"One row per input ID?"}
    V1 --> V2{"Allowed action and type?"}
    V2 --> V3{"Confidence is finite in 0..1?"}
    V3 --> V4{"Evidence IDs exist or none?"}
    V4 -->|"all pass"| O["Write output.csv"]
    V1 -->|"fail"| E["Raise validation error"]
    V2 -->|"fail"| E
    V3 -->|"fail"| E
    V4 -->|"fail"| E
```

Quality is checked at three levels:

1. Unit tests cover deterministic safety, evidence validation, and cache
   persistence.
2. The solved sample set reports action/type accuracy plus slices by
   conversation and expected type.
3. Final-output validation verifies all submission constraints before writing.

## Operational runbook

```bash
# Local Ollama run
ROUTER_LLM_PROVIDER=ollama .venv/bin/python -m code.main \
  --dataset-dir dataset --output dataset/output.csv

# Verify a provider before a long run
.venv/bin/python -m code.main --check-config --provider ollama

# Evaluate against solved examples
ROUTER_LLM_PROVIDER=ollama .venv/bin/python -m code.evaluation.main \
  --dataset-dir dataset --provider ollama
```

## Current limits and next work

- Retrieval is lexical and relationship-aware; add embeddings only after the
  current evidence audit shows a measurable need.
- The cache makes local runs resumable, but the provider is intentionally
  sequential to remain within the local 3–4 GB model budget.
- Solved-sample type accuracy remains weaker than action accuracy. Prioritize
  type-specific tuning for group messages and promotions, as recorded in
  `DECISIONS.md`.
