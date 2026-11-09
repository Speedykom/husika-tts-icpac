"""Tests for the HUSIKA TTS service — Sprint 1."""

import pytest
from fastapi.testclient import TestClient

from tts_service.api.server import app
from tts_service.engines.engine_router import EngineRouter
from tts_service.engines.espeak_engine import ESPEAK_VOICE_MAP, EspeakEngine
from tts_service.engines.mms_engine import MMS_MODELS, MmsEngine

client = TestClient(app)


# ── Health & metadata ──────────────────────────────────────────


class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_languages_returns_dict(self):
        r = client.get("/languages")
        assert r.status_code == 200
        langs = r.json()
        assert isinstance(langs, dict)
        assert "swa" in langs

    def test_languages_include_engine_availability(self):
        r = client.get("/languages")
        swa = r.json()["swa"]
        assert "engines_available" in swa


# ── Engine router ──────────────────────────────────────────────


class TestEngineRouter:
    def test_selects_engine_for_known_language(self):
        router = EngineRouter()
        engine, code = router.select_engine("swa")
        assert code == "swa"
        assert engine.name in ("mms", "espeak")

    def test_falls_back_to_english_for_unknown_language(self):
        router = EngineRouter()
        engine, code = router.select_engine("zzz")
        assert code == "en"
        assert engine.supports_language("en")

    def test_fallback_when_preferred_unavailable(self):
        """If the preferred engine doesn't support the lang, router should fall back."""
        router = EngineRouter()
        # Amharic has no espeak support, preferred is mms → should get mms
        engine, _ = router.select_engine("amh")
        assert engine.name == "mms"


# ── eSpeak engine ─────────────────────────────────────────────


class TestEspeakEngine:
    def test_supports_mapped_languages(self):
        engine = EspeakEngine()
        if engine._binary is None:
            pytest.skip("espeak-ng not installed")
        for code in ESPEAK_VOICE_MAP:
            assert engine.supports_language(code)

    def test_rejects_unmapped_language(self):
        engine = EspeakEngine()
        assert not engine.supports_language("amh")

    @pytest.mark.asyncio
    async def test_synthesize_english(self):
        engine = EspeakEngine()
        if engine._binary is None:
            pytest.skip("espeak-ng not installed")
        result = await engine.synthesize("hello world", "en")
        assert result["engine"] == "espeak"
        assert result["format"] == "wav"
        assert len(result["audio_base64"]) > 100


# ── MMS engine ────────────────────────────────────────────────


class TestMmsEngine:
    def test_supports_mapped_languages(self):
        engine = MmsEngine()
        for code in MMS_MODELS:
            assert engine.supports_language(code)

    def test_rejects_unmapped_language(self):
        engine = MmsEngine()
        assert not engine.supports_language("zzz")


# ── TTS endpoint ─────────────────────────────────────────────


class TestTTSEndpoint:
    def test_rejects_empty_text(self):
        r = client.post(
            "/tts",
            json={
                "text": "",
                "lang_code": "en",
            },
        )
        assert r.status_code == 422

    def test_falls_back_to_english_for_unknown_language(self):
        r = client.post(
            "/tts",
            json={
                "text": "hello",
                "lang_code": "zzz",
            },
        )
        assert r.status_code == 200
        assert r.json()["lang_code"] == "zzz"

    def test_rejects_invalid_speed(self):
        r = client.post(
            "/tts",
            json={
                "text": "hello",
                "lang_code": "en",
                "speed": 5.0,
            },
        )
        assert r.status_code == 422


# ── Karamojong VITS engine ────────────────────────────────────


class TestKaramojongVitsEngine:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Ngitunga daadang", "ŋituŋa daadaŋ"),
            ("ng'aren  ng’ina\n kalo", "ŋaren ŋina kalo"),
            # U+02BC is not kept by the training rules, so it splits the word
            ("ngʼina", "ŋ ina"),
            ("Ethiopia, chicha!", "etiopia cica"),
            ("fazvix qñh", "paswiks kny"),
            ("nngolo nngitunga", "ŋolo ŋituŋa"),
            ("akidindinnga", "akidindinŋa"),
            ("a-b d’e 'x- -", "a-b d'e ks"),
            ("?!", ""),
        ],
    )
    def test_normalise_text(self, raw, expected):
        from tts_service.engines.karamojong_vits_engine import normalise_text

        assert normalise_text(raw) == expected

    def test_supports_only_configured_codes(self, monkeypatch):
        from tts_service.engines import karamojong_vits_engine as mod

        monkeypatch.setattr(
            mod, "KARAMOJONG_MODELS", {"kdj-k05": {"repo": "o/r", "speaker": None}}
        )
        engine = mod.KaramojongVitsEngine()
        assert engine.supports_language("kdj-k05")
        assert not engine.supports_language("kdj")

    @pytest.mark.asyncio
    async def test_rejects_unconfigured_language(self, monkeypatch):
        from tts_service.engines import karamojong_vits_engine as mod

        monkeypatch.setattr(mod, "KARAMOJONG_MODELS", {})
        with pytest.raises(ValueError):
            await mod.KaramojongVitsEngine().synthesize("ngitunga", "kdj-k05")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["?! -", "ngitunga 12"])
    async def test_rejects_empty_text_and_digits(self, monkeypatch, text):
        from tts_service.engines import karamojong_vits_engine as mod

        monkeypatch.setattr(
            mod, "KARAMOJONG_MODELS", {"kdj-k05": {"repo": "o/r", "speaker": None}}
        )
        with pytest.raises(ValueError):
            await mod.KaramojongVitsEngine().synthesize(text, "kdj-k05")

    @pytest.mark.asyncio
    async def test_synthesize_with_stub_model(self, monkeypatch):
        """Full path with a stub single-speaker model: speed and WAV encoding."""
        import base64

        import numpy as np
        import torch

        from tts_service.engines import karamojong_vits_engine as mod

        monkeypatch.setattr(
            mod, "KARAMOJONG_MODELS", {"kdj-k05": {"repo": "o/r", "speaker": None}}
        )
        seen = {}

        class StubTokenizer:
            def text_to_ids(self, text):
                seen["text"] = text
                return [1, 2, 3]

        class StubModel:
            speakers = ["K05"]
            tokenizer = StubTokenizer()
            config = type(
                "C",
                (),
                {"model_args": type("A", (), {"use_speaker_embedding": False})()},
            )()

            def inference(self, x, aux_input):
                seen["aux_keys"] = sorted(aux_input)
                seen["length_scale"] = self.length_scale
                return {"model_outputs": torch.zeros(1, 1, 1600)}

        engine = mod.KaramojongVitsEngine()
        engine._models["kdj-k05"] = StubModel()

        result = await engine.synthesize("Ngitunga daadang", "kdj-k05", speed=2.0)

        assert result["engine"] == "karamojong_vits"
        assert result["format"] == "wav"
        assert result["sample_rate"] == 16000
        wav = base64.b64decode(result["audio_base64"])
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
        assert seen["text"] == "ŋituŋa daadaŋ"
        assert seen["aux_keys"] == ["x_lengths"]  # single speaker: no speaker id
        assert seen["length_scale"] == pytest.approx(0.5)
        assert np.frombuffer(wav[44:], dtype=np.int16).shape[0] == 1600


class TestKaramojongLanguages:
    def test_karamojong_languages_listed_with_engine(self):
        langs = client.get("/languages").json()
        assert langs["kdj-k05"]["preferred_engine"] == "karamojong_vits"
        assert langs["kdj"]["preferred_engine"] != "karamojong_vits"
