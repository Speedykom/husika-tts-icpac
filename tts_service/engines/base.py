"""Base class for TTS engines."""

from abc import ABC, abstractmethod


class TTSEngine(ABC):
    """Abstract base class that all TTS engines must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier (e.g. 'espeak', 'mms')."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language_code: str,
        speed: float = 1.0,
    ) -> dict:
        """Synthesize text to audio.

        Returns dict with keys: audio_base64, format, sample_rate, engine.
        """

    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """Check if this engine supports the given language code."""
