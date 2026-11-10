"""Karamojong VITS engine for the in-house MMS-kdj fine-tunes (ISO 639-3 ``kdj``).

The checkpoints are Coqui VITS models whose text encoder uses the MMS (fairseq)
character vocabulary of ``facebook/mms-tts-kdj``, so the stock ``Synthesizer``
path of the generic Coqui engine cannot load them. Loading follows the
``karamojong_tts.py`` loader published in each model repo, and the text rules
are ported from its ``karamojong_text.py`` (the repo code itself is not
imported, so no downloaded Python is executed).

Checkpoints are served side by side, each under its own language code:

* K05: single speaker K05, fine-tuned from ``facebook/mms-tts-kdj``

They are supplied at runtime as a JSON object in the ``KARAMOJONG_MODELS`` env
var so private repo ids stay out of the source tree::

    {"kdj-k05": "org/karamojong-k05"}

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
from .toposa_vits_engine import _parse_models

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Only these files are needed for inference; skips samples, eval and the README.
REPO_FILES = [
    "model.pth",
    "config.json",
    "mms_vocab.txt",
    "mms_config.json",
    "karamojong_run.json",
]

KARAMOJONG_MODELS: dict[str, dict] = _parse_models(
    os.environ.get("KARAMOJONG_MODELS", "")
)

# Text rules from karamojong_text.py in the model repo (provisional until a native
# speaker confirms them). The kdj base vocabulary is kept unchanged, so every
# letter outside it is mapped to the nearest base letter.
_TYPOS = {"nngolo": "ngolo", "nngitunga": "ngitunga"}
# Order matters: ng' before ng.
_DIGRAPHS = [("ng'", "ŋ"), ("ngʼ", "ŋ"), ("ng", "ŋ"), ("th", "t"), ("ch", "c")]
_OOV_MAP = {"f": "p", "v": "w", "z": "s", "q": "k", "x": "ks", "ñ": "ny", "h": ""}


def normalise_text(text: str) -> str:
    """Normalise Karamojong text exactly as during training.

    Lowercase letters and single spaces are kept, plus apostrophes and hyphens
    inside words; punctuation becomes a space. ``ng``/``ng'`` become ``ŋ``,
    ``th`` becomes ``t``, ``ch`` becomes ``c``, and loanword letters go to the
    nearest kdj letter (``f→p v→w z→s q→k x→ks ñ→ny``, ``h`` dropped).
    """
    s = unicodedata.normalize("NFC", text).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("–", "-").replace("\xa0", " ")
    s = re.sub(r"[^a-zñŋ' \-]", " ", s)
    s = re.sub(r"(?<![a-zñŋ])['\-]|['\-](?![a-zñŋ])", " ", s)
    t = " ".join(_TYPOS.get(w, w) for w in s.split())
    for src, dst in _DIGRAPHS:
        t = t.replace(src, dst)
    return "".join(_OOV_MAP.get(c, c) for c in t)


class KaramojongVitsEngine(TTSEngine):
    """Synthesize Karamojong speech with the fine-tuned VITS checkpoints."""

    name = "karamojong_vits"

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        # One synthesis at a time: inference settings (speed, speaker) live on the
        # shared model object, and the RNG seed is process-global.
        self._lock = threading.Lock()

    def supports_language(self, language_code: str) -> bool:
        return language_code in KARAMOJONG_MODELS

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

        # The checkpoint is a pickle from our own organisation's private repo, so
        # full unpickling is required.
        state = torch.load(
            os.path.join(repo_dir, "model.pth"), map_location="cpu", weights_only=False
        )["model"]
        model.load_state_dict(state, strict=True)

        model.speakers = read_json("karamojong_run.json").get("speakers") or []
        return model.eval()

    def _load_model(self, language_code: str) -> None:
        """Download the repo into the HF cache and load the checkpoint."""
        from huggingface_hub import snapshot_download

        repo_id = KARAMOJONG_MODELS[language_code]["repo"]
        logger.info(f"Loading Karamojong VITS model: {repo_id}")
        # snapshot_download honours HF_TOKEN (private repos) and HF_HOME (cache volume).
        repo_dir = snapshot_download(repo_id, allow_patterns=REPO_FILES)
        self._models[language_code] = self._load_checkpoint(repo_dir)

    async def synthesize(
        self,
        text: str,
        language_code: str,
        speed: float = 1.0,
    ) -> dict:
        if language_code not in KARAMOJONG_MODELS:
            raise ValueError(
                f"Karamojong VITS does not support language: {language_code}"
            )

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

        # The model was not trained on digits and no verified number reading exists,
        # so reject them (as the published loader does) instead of dropping them.
        if re.search(r"\d", text):
            raise ValueError("Write numbers out in words; the model has no digits")
        clean = normalise_text(text)
        if not clean:
            raise ValueError("Text has no Karamojong letters to synthesize")

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
                speaker = KARAMOJONG_MODELS[language_code]["speaker"] or names[0]
                if speaker not in names:
                    raise ValueError(f"Unknown speaker {speaker!r}; available: {names}")
                aux["speaker_ids"] = torch.LongTensor([names.index(speaker)])

            torch.manual_seed(1234)  # reproducible output, as in the published loader
            with torch.no_grad():
                out = model.inference(x, aux_input=aux)["model_outputs"]

        waveform = out.squeeze().float().cpu().numpy().astype(np.float32)
        return CoquiVitsEngine._to_wav(waveform, SAMPLE_RATE)
