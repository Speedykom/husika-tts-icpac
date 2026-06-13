"""Coqui VITS TTS engine for in-house fine-tuned models.

Coqui VITS checkpoints (.pth + config.json) are not weight-compatible with the
HuggingFace VitsModel implementation, so they need a separate engine that goes
through the Coqui Synthesizer API.
"""

import asyncio
import base64
import io
import json
import logging
import os

import numpy as np

from .base import TTSEngine

logger = logging.getLogger(__name__)

# Maps our language codes to Coqui-format HuggingFace repo IDs. These are custom
# fine-tuned models, so the mapping is supplied at runtime as a JSON object
# (lang_code -> repo id) via the COQUI_MODELS env var, keeping private repo ids
# out of the source tree.
COQUI_MODELS: dict[str, str] = json.loads(os.environ.get("COQUI_MODELS", "{}"))

# Filename inside each repo. The team uploads multiple .pth checkpoints
# (best_model.pth + intermediate checkpoints); best_model.pth is the inference target.
COQUI_CHECKPOINT_FILENAME = "best_model.pth"


class CoquiVitsEngine(TTSEngine):
    """Synthesize speech using Coqui-format VITS checkpoints."""

    name = "coqui_vits"

    def __init__(self) -> None:
        self._synthesizers: dict[str, object] = {}

    def supports_language(self, language_code: str) -> bool:
        return language_code in COQUI_MODELS

    def _load_model(self, language_code: str) -> None:
        """Download the Coqui repo into the HF cache and instantiate a Synthesizer."""
        from huggingface_hub import snapshot_download
        from TTS.utils.synthesizer import Synthesizer

        repo_id = COQUI_MODELS[language_code]
        logger.info(f"Loading Coqui VITS model: {repo_id}")

        # snapshot_download honors HF_TOKEN env var for private repos and
        # caches under HF_HOME (the same volume MMS uses).
        repo_dir = snapshot_download(repo_id)
        synthesizer = Synthesizer(
            tts_checkpoint=f"{repo_dir}/{COQUI_CHECKPOINT_FILENAME}",
            tts_config_path=f"{repo_dir}/config.json",
        )
        self._synthesizers[language_code] = synthesizer

    async def synthesize(
        self,
        text: str,
        language_code: str,
        speed: float = 1.0,
    ) -> dict:
        if language_code not in COQUI_MODELS:
            raise ValueError(f"Coqui VITS does not support language: {language_code}")

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
        if language_code not in self._synthesizers:
            self._load_model(language_code)

        synthesizer = self._synthesizers[language_code]
        waveform = np.asarray(synthesizer.tts(text), dtype=np.float32)
        sample_rate = synthesizer.output_sample_rate

        if speed != 1.0 and speed > 0:
            original_len = len(waveform)
            new_len = int(original_len / speed)
            indices = np.linspace(0, original_len - 1, new_len)
            waveform = np.interp(indices, np.arange(original_len), waveform)

        return self._to_wav(waveform, sample_rate), sample_rate

    @staticmethod
    def _to_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
        """Encode a float32 numpy waveform as a 16-bit PCM WAV byte string."""
        import struct

        waveform = np.clip(waveform, -1.0, 1.0)
        pcm_bytes = (waveform * 32767).astype(np.int16).tobytes()

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
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<H", 1))
        buf.write(struct.pack("<H", num_channels))
        buf.write(struct.pack("<I", sample_rate))
        buf.write(struct.pack("<I", byte_rate))
        buf.write(struct.pack("<H", block_align))
        buf.write(struct.pack("<H", bits_per_sample))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(pcm_bytes)

        return buf.getvalue()
