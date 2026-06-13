"""Router to select the appropriate TTS engine based on language configuration."""

from __future__ import annotations

import logging

import yaml

from .base import TTSEngine
from .coqui_vits_engine import CoquiVitsEngine
from .espeak_engine import EspeakEngine
from .mms_engine import MmsEngine

logger = logging.getLogger(__name__)


class EngineRouter:
    """Routes synthesis requests to eSpeak NG or MMS based on language config."""

    def __init__(self, config_path: str = "languages/languages.yaml"):
        self.config_path = config_path
        self.language_config: dict = self._load_config()

        self.engines: dict[str, TTSEngine] = {
            "espeak": EspeakEngine(),
            "mms": MmsEngine(),
            "coqui_vits": CoquiVitsEngine(),
        }

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config not found: {self.config_path}")
            return {}

    FALLBACK_LANGUAGE = "en"

    def select_engine(self, language_code: str) -> tuple[TTSEngine, str]:
        """Pick the best engine for a language.

        Returns (engine_instance, language_code) — the language_code is passed
        through so the engine knows which language to synthesize.

        Falls back through: preferred engine -> mms -> espeak.
        Falls back to English for unsupported language codes.
        """
        lang_info = self.language_config.get("languages", {}).get(language_code)
        if not lang_info:
            logger.warning(
                f"Unsupported language '{language_code}', falling back to English"
            )
            return self.select_engine(self.FALLBACK_LANGUAGE)

        preferred = lang_info.get("preferred_engine", "mms")

        # Try preferred engine first
        preferred_engine = self.engines.get(preferred)
        if preferred_engine and preferred_engine.supports_language(language_code):
            return preferred_engine, language_code

        # Fallback chain: mms -> espeak
        for fallback in ("mms", "espeak"):
            engine = self.engines[fallback]
            if fallback != preferred and engine.supports_language(language_code):
                logger.info(f"Falling back to {fallback} for {language_code}")
                return engine, language_code

        logger.warning(
            f"No engine available for '{language_code}', falling back to English"
        )
        return self.select_engine(self.FALLBACK_LANGUAGE)

    def get_language_info(self) -> dict:
        """Return the language config enriched with actual engine availability."""
        languages = self.language_config.get("languages", {})
        result = {}
        for code, info in languages.items():
            result[code] = {
                **info,
                "engines_available": [
                    name
                    for name, engine in self.engines.items()
                    if engine.supports_language(code)
                ],
            }
        return result
