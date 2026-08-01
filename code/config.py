"""Non-secret configuration and deterministic provider resolution."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # The rules-only runner should still start without extras.
    load_dotenv = None


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    dataset_dir: Path
    output_path: Path
    provider: str
    openai_model: str
    ollama_base_url: str
    ollama_model: str
    media_mode: str
    timeout_seconds: float
    max_evidence: int
    temperature: float
    seed: int
    whisper_model: str
    whisper_compute_type: str
    cache_path: Path

    @classmethod
    def from_environment(cls, dataset_dir: Optional[str] = None,
                         output_path: Optional[str] = None,
                         provider: Optional[str] = None) -> "Settings":
        if load_dotenv:
            load_dotenv(override=False)
        chosen_provider = (provider or _env("ROUTER_LLM_PROVIDER", "auto")).lower()
        if chosen_provider not in {"auto", "openai", "ollama"}:
            raise ValueError("ROUTER_LLM_PROVIDER must be auto, openai, or ollama")
        data = Path(dataset_dir or _env("ROUTER_DATASET_DIR", "dataset"))
        return cls(
            dataset_dir=data,
            output_path=Path(output_path or _env("ROUTER_OUTPUT_PATH", str(data / "output.csv"))),
            provider=chosen_provider,
            openai_model=_env("OPENAI_MODEL", "gpt-4.1-mini"),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=_env("OLLAMA_MODEL", "qwen2.5vl:3b"),
            media_mode=_env("ROUTER_MEDIA_MODE", "auto").lower(),
            timeout_seconds=float(_env("ROUTER_REQUEST_TIMEOUT_SECONDS", "60")),
            max_evidence=max(0, min(5, int(_env("ROUTER_MAX_RETRIEVED_EVIDENCE", "3")))),
            temperature=float(_env("ROUTER_TEMPERATURE", "0")),
            seed=int(_env("ROUTER_SEED", "42")),
            whisper_model=_env("ROUTER_WHISPER_MODEL", "tiny"),
            whisper_compute_type=_env("ROUTER_WHISPER_COMPUTE_TYPE", "int8"),
            cache_path=Path(_env("ROUTER_CACHE_PATH", ".router-cache/router.sqlite")),
        )

    def resolved_provider(self) -> str:
        """Select a provider; the classifier performs a required startup preflight."""
        if self.provider != "auto":
            return self.provider
        return "openai" if _env("OPENAI_API_KEY") else "ollama"
