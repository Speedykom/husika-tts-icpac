"""Pokot VITS engine for the in-house MMS fine-tunes (Pokot, ISO 639-3 ``pko``).

The checkpoints are Coqui VITS *training* checkpoints (generator plus a
training-only discriminator) whose text encoder uses an extended MMS (fairseq)
character vocabulary, so the stock ``Synthesizer`` path of the generic Coqui
engine cannot load them. The loading recipe follows the "How to use" section of
the model cards.

Three checkpoints are served side by side, each under its own language code:

* P1: single speaker PK03, fine-tuned from ``facebook/mms-tts-kdj``
* P2: single speaker PK03, fine-tuned from ``facebook/mms-tts-kpz``
* P3: five speakers (PK01-PK05), fine-tuned from ``facebook/mms-tts-kdj``

They are supplied at runtime as a JSON object in the ``POKOT_MODELS`` env var so
private repo ids stay out of the source tree::

    {"pko-p1": "org/pokot-p1",
     "pko-p3": {"repo": "org/pokot-p3", "speaker": "PK03"}}

A value is either a repo id or an object with ``repo`` and an optional default
``speaker`` (only meaningful for the multi-speaker checkpoint).
"""

import asyncio
import base64
import glob
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

# Only these files are needed for inference; skips charts, samples and the README.
# ``speakers.pth`` exists for the multi-speaker checkpoint only.
REPO_FILES = [
    "checkpoint_*.pth",
    "config.json",
    "mms_vocab.txt",
    "mms_config.json",
    "pokot_run.json",
    "speakers.pth",
]

POKOT_MODELS: dict[str, dict] = _parse_models(os.environ.get("POKOT_MODELS", ""))

# Pokot digraphs are written as single symbols, as during training.
_DIGRAPHS = {"ng": "ŋ", "ch": "c", "gh": "ɣ", "sh": "ʃ", "th": "θ"}
_DIGRAPH_RE = re.compile("|".join(_DIGRAPHS))


def normalise_text(text: str) -> str:
    """Normalise Pokot text as during training.

    Lowercase letters, spaces, hyphens and apostrophes are kept; everything else
    (digits, punctuation) is dropped. The digraphs ``ng ch gh sh th`` become the
    single symbols ``ŋ c ɣ ʃ θ`` of the extended model vocabulary.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^a-zŋɣʃθ\-' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _DIGRAPH_RE.sub(lambda m: _DIGRAPHS[m.group(0)], text)


class PokotVitsEngine(TTSEngine):
    """Synthesize Pokot speech with the fine-tuned VITS checkpoints."""

    name = "pokot_vits"

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        # One synthesis at a time: inference settings (speed, speaker) live on the
        # shared model object, and the RNG seed is process-global.
        self._lock = threading.Lock()

    def supports_language(self, language_code: str) -> bool:
        return language_code in POKOT_MODELS

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
        # full unpickling is required. ``disc.*`` is the training-only discriminator.
        (checkpoint,) = glob.glob(os.path.join(repo_dir, "checkpoint_*.pth"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        state = {k: v for k, v in state.items() if not k.startswith("disc.")}
        model.disc = None
        model.load_state_dict(state, strict=True)

        # Speaker name -> id. Prefer the table saved with the checkpoint (it is the
        # id order used in training); fall back to the order in the run metadata.
        speakers_file = os.path.join(repo_dir, "speakers.pth")
        if os.path.exists(speakers_file):
            table = torch.load(speakers_file, map_location="cpu", weights_only=False)
            model.speaker_ids = {name: int(i) for name, i in table.items()}
        else:
            names = read_json("pokot_run.json").get("speakers") or []
            model.speaker_ids = {name: i for i, name in enumerate(names)}
        return model.eval()

    def _load_model(self, language_code: str) -> None:
        """Download the repo into the HF cache and load the checkpoint."""
        from huggingface_hub import snapshot_download

        repo_id = POKOT_MODELS[language_code]["repo"]
        logger.info(f"Loading Pokot VITS model: {repo_id}")
        # snapshot_download honours HF_TOKEN (private repos) and HF_HOME (cache volume).
        repo_dir = snapshot_download(repo_id, allow_patterns=REPO_FILES)
        self._models[language_code] = self._load_checkpoint(repo_dir)

    async def synthesize(
        self,
        text: str,
        language_code: str,
        speed: float = 1.0,
    ) -> dict:
        if language_code not in POKOT_MODELS:
            raise ValueError(f"Pokot VITS does not support language: {language_code}")

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
            raise ValueError("Text has no Pokot letters to synthesize")

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
                speaker = POKOT_MODELS[language_code]["speaker"] or next(
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
