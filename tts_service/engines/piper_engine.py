"""Piper TTS engine — ONNX-based neural synthesis via piper-tts."""

import asyncio
import base64
import io
import logging
import os
import wave
from pathlib import Path

from .base import TTSEngine

logger = logging.getLogger(__name__)

PIPER_MODELS_DIR = Path(os.environ.get("PIPER_MODELS_DIR", "data/piper-models"))

PIPER_MODELS: dict[str, str] = {
    "en": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "fr": "fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
    "swa": "sw/sw_CD/lanfrica/medium/sw_CD-lanfrica-medium.onnx",
    "ara": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
}


class PiperEngine(TTSEngine):
    """Synthesize speech using Piper ONNX voice models from rhasspy/piper-voices."""

    name = "piper"

    def __init__(self) -> None:
        self._voices: dict = {}

    def supports_language(self, language_code: str) -> bool:
        return language_code in PIPER_MODELS

    def _ensure_model(self, language_code: str) -> Path:
        """Download .onnx + .onnx.json if not cached; return path to .onnx file."""
        from huggingface_hub import hf_hub_download

        hf_path = PIPER_MODELS[language_code]
        onnx_path = PIPER_MODELS_DIR / hf_path
        if not onnx_path.exists():
            PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading Piper model for %s: %s", language_code, hf_path)
            hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=hf_path,
                local_dir=str(PIPER_MODELS_DIR),
            )
            hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=hf_path + ".json",
                local_dir=str(PIPER_MODELS_DIR),
            )
        return onnx_path

    def _load_voice(self, language_code: str) -> None:
        from piper.voice import PiperVoice

        onnx_path = self._ensure_model(language_code)
        logger.info("Loading Piper voice: %s", onnx_path)
        # PiperVoice.load auto-discovers the .onnx.json config alongside the model.
        self._voices[language_code] = PiperVoice.load(str(onnx_path), use_cuda=False)

    async def synthesize(
        self,
        text: str,
        language_code: str,
        speed: float = 1.0,
    ) -> dict:
        if language_code not in PIPER_MODELS:
            raise ValueError(f"Piper does not support language: {language_code}")

        loop = asyncio.get_event_loop()
        audio_bytes, sample_rate = await loop.run_in_executor(
            None, self._run_synthesis, text, language_code, speed
        )
        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "format": "wav",
            "sample_rate": sample_rate,
            "engine": self.name,
        }

    def _run_synthesis(
        self, text: str, language_code: str, speed: float
    ) -> tuple[bytes, int]:
        import numpy as np
        from piper.voice import SynthesisConfig

        if language_code not in self._voices:
            self._load_voice(language_code)

        voice = self._voices[language_code]
        # length_scale > 1 is slower; invert speed so speed > 1 goes faster.
        syn_config = SynthesisConfig(length_scale=1.0 / speed)

        buf = io.BytesIO()
        sample_rate = voice.config.sample_rate
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # int16
            wav_file.setframerate(sample_rate)
            for chunk in voice.synthesize(text, syn_config):
                pcm = (chunk.audio_float_array * 32767).clip(-32768, 32767).astype(np.int16)
                wav_file.writeframes(pcm.tobytes())

        return buf.getvalue(), sample_rate
