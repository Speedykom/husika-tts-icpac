"""Turkana VITS engine for the in-house MMS fine-tunes (Turkana, ISO 639-3 ``tuv``).

The checkpoints are Coqui VITS generators fine-tuned from ``facebook/mms-tts-kdj``
(Turkana has no MMS voice of its own, so the closely related Karamojong voice is
the base). Their text encoder uses an extended 33-symbol MMS (fairseq) character
vocabulary, so the stock ``Synthesizer`` path of the generic Coqui engine cannot
load them. The loading recipe follows the "How to use" section of the model cards.

Two checkpoints are served side by side, each under its own language code:

* ``tuv-open``: trained on the open ``turkana-speech-dataset-v2`` (one voice, ``GRN``)
* ``tuv-col``: trained on the collected recordings (14 voices, ``T01``-``T14``)

They are supplied at runtime as a JSON object in the ``TURKANA_MODELS`` env var so
private repo ids stay out of the source tree::

    {"tuv-open": "org/turkana-open",
     "tuv-col": {"repo": "org/turkana-collected", "speaker": "T05"}}

A value is either a repo id or an object with ``repo`` and an optional default
``speaker`` (the first voice of the checkpoint is used when it is left out).
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
from .toposa_vits_engine import _parse_models

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Only these files are needed for inference; skips samples, metrics and the README.
REPO_FILES = [
    "model.pth",
    "config.json",
    "mms_vocab.txt",
    "mms_config.json",
    "speakers.pth",
    "turkana_run.json",
]

TURKANA_MODELS: dict[str, dict] = _parse_models(os.environ.get("TURKANA_MODELS", ""))

# Turkana digraphs are written as single symbols, as during training. The order
# matters and matches the training code: sh, th, ch, then ng.
_DIGRAPHS = (("sh", "ʃ"), ("th", "θ"), ("ch", "c"), ("ng", "ŋ"))
_DROP = re.compile(r"[.,;:?!()/%…\"“”\[\]]")


def normalise_text(text: str) -> str:
    """Normalise Turkana text exactly as during training.

    Lowercase; the punctuation ``. , ; : ? ! ( ) / % …``, quotes and brackets is
    removed; curly apostrophes become ``'``; spaces collapse. The digraphs
    ``sh th ch ng`` become the single symbols ``ʃ θ c ŋ`` of the extended model
    vocabulary. Text with digits is rejected: no digit sentences were in the
    training data and there is no verified way to read numbers aloud.
    """
    if any(c.isdigit() for c in text):
        raise ValueError("Text contains digits; the Turkana models cannot read numbers")
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = _DROP.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for src, dst in _DIGRAPHS:
        text = text.replace(src, dst)
    return text


class TurkanaVitsEngine(TTSEngine):
    """Synthesize Turkana speech with the fine-tuned VITS checkpoints."""

    name = "turkana_vits"

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        # One synthesis at a time: inference settings (speed, speaker) live on the
        # shared model object, and the RNG seed is process-global.
        self._lock = threading.Lock()

    def supports_language(self, language_code: str) -> bool:
        return language_code in TURKANA_MODELS

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

        # Speaker name -> id, in the order used in training. The speaker count must
        # be set before the model is built: coqui-tts 0.27.x only creates the
        # speaker-embedding layer when ``num_speakers`` is known up front (the saved
        # config has it unset and points ``speakers_file`` at a training-time path).
        table = torch.load(
            os.path.join(repo_dir, "speakers.pth"),
            map_location="cpu",
            weights_only=False,
        )
        speaker_ids = {name: int(i) for name, i in table.items()}
        config.model_args.num_speakers = len(speaker_ids)

        model = Vits(config)
        vocab_file = os.path.join(repo_dir, "mms_vocab.txt")
        if hasattr(FairseqVocab, "init_from_vocab_file"):  # coqui-tts >= 0.28
            vocab = FairseqVocab.init_from_vocab_file(Path(vocab_file))
        else:  # coqui-tts 0.27.x reads the file in the constructor
            vocab = FairseqVocab(vocab_file)
        # The extended vocabulary is larger than the base model's embedding table.
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

        # The checkpoint is a pickle from our own organisation's private repo, so
        # full unpickling is required. The release holds the generator only.
        state = torch.load(
            os.path.join(repo_dir, "model.pth"), map_location="cpu", weights_only=False
        )["model"]
        model.disc = None
        model.load_state_dict(state, strict=True)

        model.speaker_ids = speaker_ids
        return model.eval()

    def _load_model(self, language_code: str) -> None:
        """Download the repo into the HF cache and load the checkpoint."""
        from huggingface_hub import snapshot_download

        repo_id = TURKANA_MODELS[language_code]["repo"]
        logger.info(f"Loading Turkana VITS model: {repo_id}")
        # snapshot_download honours HF_TOKEN (private repos) and HF_HOME (cache volume).
        repo_dir = snapshot_download(repo_id, allow_patterns=REPO_FILES)
        self._models[language_code] = self._load_checkpoint(repo_dir)

    async def synthesize(
        self,
        text: str,
        language_code: str,
        speed: float = 1.0,
    ) -> dict:
        if language_code not in TURKANA_MODELS:
            raise ValueError(f"Turkana VITS does not support language: {language_code}")

        loop = asyncio.get_running_loop()
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
            raise ValueError("Text has no Turkana letters to synthesize")

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
                ids_by_name = model.speaker_ids
                speaker = TURKANA_MODELS[language_code]["speaker"] or next(
                    iter(ids_by_name)
                )
                if speaker not in ids_by_name:
                    raise ValueError(
                        f"Unknown speaker {speaker!r}; available: {list(ids_by_name)}"
                    )
                aux["speaker_ids"] = torch.LongTensor([ids_by_name[speaker]])

            torch.manual_seed(1234)  # reproducible output
            with torch.no_grad():
                out = model.inference(x, aux_input=aux)["model_outputs"]

        waveform = out.squeeze().float().cpu().numpy().astype(np.float32)
        return CoquiVitsEngine._to_wav(waveform, SAMPLE_RATE)
