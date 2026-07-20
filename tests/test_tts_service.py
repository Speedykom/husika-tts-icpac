"""Tests for the HUSIKA TTS service — Sprint 1."""

import uuid

import pytest
from fastapi.testclient import TestClient

from tts_service.api.server import app
from tts_service.engines.engine_router import EngineRouter
from tts_service.engines.espeak_engine import ESPEAK_VOICE_MAP, EspeakEngine
from tts_service.engines.mms_engine import MMS_MODELS, MmsEngine
from tts_service.ratings.store import RatingsStore

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


# ── Ratings persistence ────────────────────────────────────────


class TestRatingsHistory:
    def test_resubmitting_keeps_all_reviews(self):
        """Re-rating the same phrase appends a new row instead of overwriting."""
        store = RatingsStore()
        phrase = f"history-test-{uuid.uuid4()}"
        store.add("tester", "en", phrase, 2, "first take")
        store.add("tester", "en", phrase, 5, "second take")

        rows = store.query(reviewer="tester", language="en", phrase=phrase)

        assert len(rows) == 2  # both reviews persisted
        # query returns newest-first, so the latest review leads
        assert [r["rating"] for r in rows] == [5, 2]
        assert rows[0]["comment"] == "second take"
