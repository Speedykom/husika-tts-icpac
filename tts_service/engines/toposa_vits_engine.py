"""Toposa VITS engine for the in-house MMS-kdj fine-tunes (Toposa, ISO 639-3 ``toq``).

The checkpoints are Coqui VITS models whose text encoder uses the MMS (fairseq)
character vocabulary, so the stock ``Synthesizer`` path used by the generic Coqui
engine cannot load them. Loading and text normalisation here follow the
``toposa_tts.py`` loader published in each model repo, so the audio matches the
reference samples shipped with the models.

Several checkpoints (different training runs) are served side by side, each under
its own language code. They are supplied at runtime as a JSON object in the
``TOPOSA_MODELS`` env var so private repo ids stay out of the source tree::

    {"toq-clear": "org/toposa-clear",
     "toq-5spk":  {"repo": "org/toposa-5speakers", "speaker": "LA"}}

A value is either a repo id or an object with ``repo`` and an optional default
``speaker`` (only meaningful for multi-speaker checkpoints).
"""

import asyncio
import base64
import json
import logging
import os
import re
import threading
import unicodedata
from pathlib import Path

import numpy as np

from .base import TTSEngine
from .coqui_vits_engine import CoquiVitsEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Only these files are needed for inference; skips charts, samples and metrics.
REPO_FILES = [
    "model.pth",
    "config.json",
    "mms_vocab.txt",
    "mms_config.json",
    "toposa_run.json",
]


def _parse_models(raw: str) -> dict[str, dict]:
    """Parse the TOPOSA_MODELS env value into ``{code: {"repo", "speaker"}}``."""
    if not raw.strip():
        return {}
    models: dict[str, dict] = {}
    for code, value in json.loads(raw).items():
        if isinstance(value, str):
            value = {"repo": value}
        models[code] = {"repo": value["repo"], "speaker": value.get("speaker")}
    return models


TOPOSA_MODELS: dict[str, dict] = _parse_models(os.environ.get("TOPOSA_MODELS", ""))


def normalise_text(text: str) -> str:
    """Normalise Toposa text exactly as during training.

    Lowercase letters, spaces, hyphens and apostrophes are kept; everything else
    is dropped, and ``ng`` is written as the single letter ``ŋ`` (the base
    model's own symbol).
    """
    text = unicodedata.normalize("NFC", text).lower().replace("ŋ", "ng")
    text = re.sub(r"[^a-z\-' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("ng", "ŋ")


class ToposaVitsEngine(TTSEngine):
    """Synthesize Toposa speech with the fine-tuned VITS checkpoints."""

    name = "toposa_vits"

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        # One synthesis at a time: inference settings (speed, speaker) live on the
        # shared model object, and the RNG seed is process-global.
        self._lock = threading.Lock()

    def supports_language(self, language_code: str) -> bool:
        return language_code in TOPOSA_MODELS

    @staticmethod
    def _load_checkpoint(repo_dir: str):
        """Build a Vits model from a downloaded repo snapshot (CPU, eval mode)."""
        import torch
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import FairseqVocab, Vits
        from TTS.tts.utils.text.cleaners import basic_cleaners
        from TTS.tts.utils.text.tokenizer import TTSTokenizer

        def read_json(name: str) -> dict:
            with open(os.path.join(repo_dir, name), encoding="utf-8") as f:
                return json.load(f)

        config = VitsConfig()
        config.load_json(os.path.join(repo_dir, "config.json"))
        mms = read_json("mms_config.json")

        model = Vits(config)
        vocab_file = os.path.join(repo_dir, "mms_vocab.txt")
        if hasattr(FairseqVocab, "init_from_vocab_file"):  # coqui-tts >= 0.28
            vocab = FairseqVocab.init_from_vocab_file(Path(vocab_file))
        else:  # coqui-tts 0.27.x reads the file in the constructor
            vocab = FairseqVocab(vocab_file)
        model.text_encoder.emb = torch.nn.Embedding(
            vocab.num_chars, config.model_args.hidden_channels
        )
        model.tokenizer = TTSTokenizer(
            use_phonemes=False,
            text_cleaner=basic_cleaners,
            characters=vocab,
            phonemizer=None,
            add_blank=bool(mms["data"].get("add_blank", True)),
            use_eos_bos=False,
        )
        model.disc = None  # discriminator is training-only and not part of the release

        # The checkpoint is a pickle from our own organisation's repo (same trust
        # level as the other Coqui models), so full unpickling is required.
        state = torch.load(
            os.path.join(repo_dir, "model.pth"), map_location="cpu", weights_only=False
        )["model"]
        model.load_state_dict(state, strict=True)

        model.speakers = read_json("toposa_run.json").get("speakers") or []
        return model.eval()

    def _load_model(self, language_code: str) -> None:
        """Download the repo into the HF cache and load the checkpoint."""
        from huggingface_hub import snapshot_download

        repo_id = TOPOSA_MODELS[language_code]["repo"]
        logger.info(f"Loading Toposa VITS model: {repo_id}")
        # snapshot_download honours HF_TOKEN (private repos) and HF_HOME (cache volume).
        repo_dir = snapshot_download(repo_id, allow_patterns=REPO_FILES)
        self._models[language_code] = self._load_checkpoint(repo_dir)

    async def synthesize(
        self,
        text: str,
        language_code: str,
        speed: float = 1.0,
    ) -> dict:
        if language_code not in TOPOSA_MODELS:
            raise ValueError(f"Toposa VITS does not support language: {language_code}")

        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None, self._run_synthesis, text, language_code, speed
        )

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "format": "wav",
            "sample_rate": SAMPLE_RATE,
            "engine": self.name,
        }

    def _run_synthesis(self, text: str, language_code: str, speed: float) -> bytes:
        import torch

        clean = normalise_text(text)
        if not clean:
            raise ValueError("Text has no Toposa letters to synthesize")

        with self._lock:
            if language_code not in self._models:
                self._load_model(language_code)
            model = self._models[language_code]

            model.inference_noise_scale = 0.667
            model.inference_noise_scale_dp = 0.8
            model.length_scale = 1.0 / speed if speed > 0 else 1.0

            ids = model.tokenizer.text_to_ids(clean)
            x = torch.LongTensor(ids).unsqueeze(0)
            aux = {"x_lengths": torch.LongTensor([x.shape[1]])}

            if model.config.model_args.use_speaker_embedding:
                names = model.speakers
                speaker = TOPOSA_MODELS[language_code]["speaker"] or names[0]
                if speaker not in names:
                    raise ValueError(f"Unknown speaker {speaker!r}; available: {names}")
                aux["speaker_ids"] = torch.LongTensor([names.index(speaker)])

            torch.manual_seed(1234)  # reproducible output, as in the published loader
            with torch.no_grad():
                out = model.inference(x, aux_input=aux)["model_outputs"]

        waveform = out.squeeze().float().cpu().numpy().astype(np.float32)
        return CoquiVitsEngine._to_wav(waveform, SAMPLE_RATE)
