"""eSpeak NG TTS engine — uses the espeak-ng CLI via subprocess."""

import asyncio
import base64
import logging
import shutil
import subprocess

from .base import TTSEngine

logger = logging.getLogger(__name__)

# Map ISO 639-3 codes used in languages.yaml to espeak-ng voice names.
# espeak-ng uses its own voice identifiers; this map covers the languages
# we support that have espeak voices available.
ESPEAK_VOICE_MAP: dict[str, str] = {
    "swa": "sw",
    "ara": "ar",
    "tir": "ti",
    "en": "en",
    "fr": "fr",
}


class EspeakEngine(TTSEngine):
    """Synthesize speech via the locally installed espeak-ng binary."""

    name = "espeak"

    def __init__(self) -> None:
        self._binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self._binary:
            logger.warning("espeak-ng binary not found — engine unavailable")

    def supports_language(self, language_code: str) -> bool:
        return language_code in ESPEAK_VOICE_MAP and self._binary is not None

    async def synthesize(
        self,
        text: str,
        language_code: str,
        speed: float = 1.0,
    ) -> dict:
        if not self._binary:
            raise RuntimeError("espeak-ng is not installed")

        voice = ESPEAK_VOICE_MAP.get(language_code)
        if not voice:
            raise ValueError(f"eSpeak does not support language: {language_code}")

        wpm = int(175 * speed)  # espeak default is 175 wpm

        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(None, self._run, text, voice, wpm)

        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 22050,
            "engine": self.name,
        }

    def _run(self, text: str, voice: str, wpm: int) -> bytes:
        """Call espeak-ng and return raw WAV bytes."""
        cmd = [
            self._binary,
            "-v",
            voice,
            "-s",
            str(wpm),
            "--stdout",
            text,
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout
