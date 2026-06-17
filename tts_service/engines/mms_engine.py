"""Meta MMS (Massively Multilingual Speech) TTS engine via HuggingFace Transformers."""

import asyncio
import base64
import io
import json
import logging
import os

import numpy as np

from .base import TTSEngine

logger = logging.getLogger(__name__)

# Maps our language codes to HuggingFace model IDs (facebook/mms-tts-{iso639-3}).
MMS_MODELS: dict[str, str] = {
    "swa": "facebook/mms-tts-swh",
    "amh": "facebook/mms-tts-amh",
    "ara": "facebook/mms-tts-ara",
    "som": "facebook/mms-tts-som",
    "orm": "facebook/mms-tts-orm",
    "tir": "facebook/mms-tts-tir",
    "lug": "facebook/mms-tts-lug",
    "kin": "facebook/mms-tts-kin",
    "rn": "facebook/mms-tts-run",
    "nue": "facebook/mms-tts-nus",
    "din": "facebook/mms-tts-dik",
    "en": "facebook/mms-tts-eng",
    "fr": "facebook/mms-tts-fra",
}

# Custom fine-tuned models override the public defaults per language. Supplied at
# runtime as a JSON object (lang_code -> repo id) via the MMS_MODEL_OVERRIDES env
# var, so private repo ids are never hard-coded in the source tree.
MMS_MODELS.update(json.loads(os.environ.get("MMS_MODEL_OVERRIDES", "{}")))


class MmsEngine(TTSEngine):
    """Synthesize speech using Meta's MMS-TTS models from HuggingFace."""

    name = "mms"

    def __init__(self) -> None:
        # Models are loaded lazily per language to avoid huge upfront cost.
        self._models: dict[str, tuple] = {}  # lang -> (model, tokenizer)

    def supports_language(self, language_code: str) -> bool:
        return language_code in MMS_MODELS

    def _load_model(self, language_code: str):
        """Download and cache the MMS-TTS model for a language."""
        from transformers import AutoTokenizer, VitsModel

        model_id = MMS_MODELS[language_code]
        logger.info(f"Loading MMS model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id)
        self._models[language_code] = (model, tokenizer)

    async def synthesize(
        self,
        text: str,
        language_code: str,
        speed: float = 1.0,
    ) -> dict:
        if language_code not in MMS_MODELS:
            raise ValueError(f"MMS does not support language: {language_code}")

        loop = asyncio.get_event_loop()
        audio_bytes, sample_rate = await loop.run_in_executor(
            None, self._run_synthesis, text, language_code, speed
        )

        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": sample_rate,
            "engine": self.name,
        }

    def _run_synthesis(
        self, text: str, language_code: str, speed: float
    ) -> tuple[bytes, int]:
        """Run model inference and return (wav_bytes, sample_rate)."""
        import torch

        if language_code not in self._models:
            self._load_model(language_code)

        model, tokenizer = self._models[language_code]

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs)

        waveform = output.waveform[0].cpu().numpy()
        sample_rate = model.config.sampling_rate

        # Apply speed adjustment by resampling
        if speed != 1.0 and speed > 0:
            original_len = len(waveform)
            new_len = int(original_len / speed)
            indices = np.linspace(0, original_len - 1, new_len)
            waveform = np.interp(indices, np.arange(original_len), waveform)

        # Convert float32 waveform to 16-bit PCM WAV
        wav_bytes = self._to_wav(waveform, sample_rate)
        return wav_bytes, sample_rate

    @staticmethod
    def _to_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
        """Encode a float32 numpy waveform as a WAV byte string."""
        import struct

        # Normalize and convert to int16
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm = (waveform * 32767).astype(np.int16)
        pcm_bytes = pcm.tobytes()

        # Build WAV header (44 bytes) + data
        buf = io.BytesIO()
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_bytes)

        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))  # chunk size
        buf.write(struct.pack("<H", 1))  # PCM format
        buf.write(struct.pack("<H", num_channels))
        buf.write(struct.pack("<I", sample_rate))
        buf.write(struct.pack("<I", byte_rate))
        buf.write(struct.pack("<H", block_align))
        buf.write(struct.pack("<H", bits_per_sample))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(pcm_bytes)

        return buf.getvalue()
