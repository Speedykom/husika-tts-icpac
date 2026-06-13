"""Pre-load MMS models on container boot.

Each MMS model is ~100 MB. Lazy-loading at request time would make the first
synthesis call take minutes per language. This script downloads (or loads from
the mounted hf_cache volume) every MMS-supported language up front, so testers
never hit a cold start.

Run as a background process from the container entrypoint — failures are
logged but do not block uvicorn from starting.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [prewarm] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES_FILE = REPO_ROOT / "languages" / "languages.yaml"


def _warm_mms(targets: list[str]) -> list[str]:
    from transformers import AutoTokenizer, VitsModel

    from tts_service.engines.mms_engine import MMS_MODELS

    failures: list[str] = []
    for code in targets:
        model_id = MMS_MODELS[code]
        try:
            t0 = time.monotonic()
            AutoTokenizer.from_pretrained(model_id)
            VitsModel.from_pretrained(model_id)
            logger.info("ok  mms %s (%s) in %.1fs", code, model_id, time.monotonic() - t0)
        except Exception as e:
            failures.append(code)
            logger.warning("fail mms %s (%s): %s", code, model_id, e)
    return failures


def _warm_coqui(targets: list[str]) -> list[str]:
    from huggingface_hub import snapshot_download

    from tts_service.engines.coqui_vits_engine import COQUI_MODELS

    failures: list[str] = []
    for code in targets:
        repo_id = COQUI_MODELS[code]
        try:
            t0 = time.monotonic()
            snapshot_download(repo_id)
            logger.info(
                "ok  coqui %s (%s) in %.1fs", code, repo_id, time.monotonic() - t0
            )
        except Exception as e:
            failures.append(code)
            logger.warning("fail coqui %s (%s): %s", code, repo_id, e)
    return failures


def main() -> int:
    from tts_service.engines.coqui_vits_engine import COQUI_MODELS
    from tts_service.engines.mms_engine import MMS_MODELS

    with LANGUAGES_FILE.open() as f:
        config = yaml.safe_load(f) or {}

    languages = config.get("languages", {})
    mms_targets = [
        code
        for code, info in languages.items()
        if info.get("mms_support") and code in MMS_MODELS
    ]
    coqui_targets = [
        code
        for code, info in languages.items()
        if info.get("coqui_vits_support") and code in COQUI_MODELS
    ]

    logger.info(
        "warming %d MMS + %d Coqui languages",
        len(mms_targets),
        len(coqui_targets),
    )

    started = time.monotonic()
    failures = _warm_mms(mms_targets) + _warm_coqui(coqui_targets)
    total = len(mms_targets) + len(coqui_targets)

    logger.info(
        "done in %.1fs — %d ok, %d failed",
        time.monotonic() - started,
        total - len(failures),
        len(failures),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
