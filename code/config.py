"""Non-secret configuration and deterministic provider resolution."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # Configuration can still report a useful setup error.
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
    max_retries: int
    retry_mode: str
    retry_base_seconds: float
    retry_max_seconds: float
    retry_jitter_seconds: float
    message_deadline_seconds: float
    run_deadline_seconds: float
    vision_provider: str
    vision_model: str
    audio_provider: str
    openai_transcription_model: str
    embedding_provider: str
    openai_embedding_model: str
    ollama_embedding_model: str
    embedding_batch_size: int

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
        media_mode = _env("ROUTER_MEDIA_MODE", "auto").lower()
        if media_mode not in {"auto", "off"}:
            raise ValueError("ROUTER_MEDIA_MODE must be auto or off")
        retry_mode = _env("ROUTER_RETRY_MODE", "exponential").lower()
        if retry_mode not in {"none", "fixed", "exponential"}:
            raise ValueError("ROUTER_RETRY_MODE must be none, fixed, or exponential")
        vision_provider = _env("ROUTER_VISION_PROVIDER", "auto").lower()
        if vision_provider not in {"auto", "none", "openai", "ollama"}:
            raise ValueError("ROUTER_VISION_PROVIDER must be auto, none, openai, or ollama")
        audio_provider = _env("ROUTER_AUDIO_PROVIDER", "auto").lower()
        if audio_provider not in {"auto", "none", "openai", "local"}:
            raise ValueError("ROUTER_AUDIO_PROVIDER must be auto, none, openai, or local")
        embedding_provider = _env("ROUTER_EMBEDDING_PROVIDER", "auto").lower()
        if embedding_provider not in {"auto", "none", "openai", "ollama"}:
            raise ValueError("ROUTER_EMBEDDING_PROVIDER must be auto, none, openai, or ollama")
        return cls(
            dataset_dir=data,
            output_path=Path(output_path or _env("ROUTER_OUTPUT_PATH", str(data / "output.csv"))),
            provider=chosen_provider,
            openai_model=_env("OPENAI_MODEL", "gpt-4.1-mini"),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=_env("OLLAMA_MODEL", "qwen2.5vl:3b"),
            media_mode=media_mode,
            timeout_seconds=float(_env("ROUTER_REQUEST_TIMEOUT_SECONDS", "60")),
            max_evidence=max(0, min(5, int(_env("ROUTER_MAX_RETRIEVED_EVIDENCE", "3")))),
            temperature=float(_env("ROUTER_TEMPERATURE", "0")),
            seed=int(_env("ROUTER_SEED", "42")),
            whisper_model=_env("ROUTER_WHISPER_MODEL", "tiny"),
            whisper_compute_type=_env("ROUTER_WHISPER_COMPUTE_TYPE", "int8"),
            cache_path=Path(_env("ROUTER_CACHE_PATH", ".router-cache/router.sqlite")),
            max_retries=max(0, min(5, int(_env("ROUTER_MAX_RETRIES", "2")))),
            retry_mode=retry_mode,
            retry_base_seconds=max(0.0, float(_env("ROUTER_RETRY_BASE_SECONDS", "0.5"))),
            retry_max_seconds=max(0.0, float(_env("ROUTER_RETRY_MAX_SECONDS", "8"))),
            retry_jitter_seconds=max(0.0, float(_env("ROUTER_RETRY_JITTER_SECONDS", "0.25"))),
            message_deadline_seconds=max(1.0, float(_env("ROUTER_MESSAGE_DEADLINE_SECONDS", "180"))),
            run_deadline_seconds=max(0.0, float(_env("ROUTER_RUN_DEADLINE_SECONDS", "0"))),
            vision_provider=vision_provider,
            vision_model=_env("ROUTER_VISION_MODEL", _env("OLLAMA_MODEL", "qwen2.5vl:3b")),
            audio_provider=audio_provider,
            openai_transcription_model=_env("OPENAI_TRANSCRIPTION_MODEL", "whisper-1"),
            embedding_provider=embedding_provider,
            openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            ollama_embedding_model=_env("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            embedding_batch_size=max(1, min(256, int(_env("ROUTER_EMBEDDING_BATCH_SIZE", "64")))),
        )

    def resolved_provider(self) -> str:
        """Select a provider; the classifier performs a required startup preflight."""
        if self.provider != "auto":
            return self.provider
        return "openai" if _env("OPENAI_API_KEY") else "ollama"

    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider
        return "openai" if _env("OPENAI_API_KEY") else "ollama"

    def resolved_vision_provider(self) -> str:
        if self.vision_provider != "auto":
            return self.vision_provider
        return "openai" if self.resolved_provider() == "openai" else "ollama"

    def resolved_audio_provider(self) -> str:
        if self.audio_provider != "auto":
            return self.audio_provider
        return "openai" if self.resolved_provider() == "openai" else "local"
