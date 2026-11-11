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


# ── Toposa VITS engine ────────────────────────────────────────


class TestToposaVitsEngine:
    def test_parse_models_accepts_string_and_object_values(self):
        from tts_service.engines.toposa_vits_engine import _parse_models

        models = _parse_models(
            '{"toq-clear": "org/clear",'
            ' "toq-5spk": {"repo": "org/five", "speaker": "LA"}}'
        )
        assert models["toq-clear"] == {"repo": "org/clear", "speaker": None}
        assert models["toq-5spk"] == {"repo": "org/five", "speaker": "LA"}

    def test_parse_models_empty_env_is_no_models(self):
        from tts_service.engines.toposa_vits_engine import _parse_models

        assert _parse_models("") == {}
        assert _parse_models("  ") == {}

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Tana Joko", "tana joko"),
            ("ngibaren  kalo\n wukor", "ŋibaren kalo wukor"),
            ("Ngakipi, koni 12!", "ŋakipi koni"),
            ("ŋibaren", "ŋibaren"),
            ("a-b d'e", "a-b d'e"),
            ("123 ?!", ""),
        ],
    )
    def test_normalise_text(self, raw, expected):
        from tts_service.engines.toposa_vits_engine import normalise_text

        assert normalise_text(raw) == expected

    def test_supports_only_configured_codes(self, monkeypatch):
        from tts_service.engines import toposa_vits_engine as mod

        monkeypatch.setattr(
            mod, "TOPOSA_MODELS", {"toq-clear": {"repo": "o/r", "speaker": None}}
        )
        engine = mod.ToposaVitsEngine()
        assert engine.supports_language("toq-clear")
        assert not engine.supports_language("toq-natural")

    @pytest.mark.asyncio
    async def test_rejects_unconfigured_language(self, monkeypatch):
        from tts_service.engines import toposa_vits_engine as mod

        monkeypatch.setattr(mod, "TOPOSA_MODELS", {})
        with pytest.raises(ValueError):
            await mod.ToposaVitsEngine().synthesize("tana joko", "toq-clear")

    @pytest.mark.asyncio
    async def test_rejects_text_without_toposa_letters(self, monkeypatch):
        from tts_service.engines import toposa_vits_engine as mod

        monkeypatch.setattr(
            mod, "TOPOSA_MODELS", {"toq-clear": {"repo": "o/r", "speaker": None}}
        )
        with pytest.raises(ValueError):
            await mod.ToposaVitsEngine().synthesize("123 !?", "toq-clear")

    @pytest.mark.asyncio
    async def test_synthesize_with_stub_model(self, monkeypatch):
        """Full path with a stub model: speaker choice, speed and WAV encoding."""
        import base64

        import numpy as np
        import torch

        from tts_service.engines import toposa_vits_engine as mod

        monkeypatch.setattr(
            mod,
            "TOPOSA_MODELS",
            {"toq-5spk": {"repo": "o/r", "speaker": "LA"}},
        )
        seen = {}

        class StubTokenizer:
            def text_to_ids(self, text):
                seen["text"] = text
                return [1, 2, 3]

        class StubModel:
            speakers = ["AP", "LA"]
            tokenizer = StubTokenizer()
            config = type(
                "C",
                (),
                {"model_args": type("A", (), {"use_speaker_embedding": True})()},
            )()

            def inference(self, x, aux_input):
                seen["speaker_ids"] = aux_input["speaker_ids"].tolist()
                seen["length_scale"] = self.length_scale
                return {"model_outputs": torch.zeros(1, 1, 1600)}

        engine = mod.ToposaVitsEngine()
        engine._models["toq-5spk"] = StubModel()

        result = await engine.synthesize("Ngibaren kalo", "toq-5spk", speed=2.0)

        assert result["engine"] == "toposa_vits"
        assert result["format"] == "wav"
        assert result["sample_rate"] == 16000
        wav = base64.b64decode(result["audio_base64"])
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
        assert seen["text"] == "ŋibaren kalo"
        assert seen["speaker_ids"] == [1]  # configured speaker "LA"
        assert seen["length_scale"] == pytest.approx(0.5)
        assert np.frombuffer(wav[44:], dtype=np.int16).shape[0] == 1600


class TestToposaLanguages:
    def test_toposa_languages_listed_with_engine(self):
        langs = client.get("/languages").json()
        for code in ("toq-clear", "toq-natural", "toq-5spk"):
            assert code in langs
            assert langs[code]["preferred_engine"] == "toposa_vits"


# ── Pokot VITS engine ─────────────────────────────────────────


class TestPokotVitsEngine:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Poru Rop", "poru rop"),
            ("ngala  achey\n kininyogh", "ŋala acey kininyoɣ"),
            ("ighesyeghut chona", "iɣesyeɣut cona"),
            ("shamba thiny", "ʃamba θiny"),
            ("ŋala", "ŋala"),
            ("a-b d'e", "a-b d'e"),
            ("Somalia, 12 !", "somalia"),
            ("123 ?!", ""),
        ],
    )
    def test_normalise_text(self, raw, expected):
        from tts_service.engines.pokot_vits_engine import normalise_text

        assert normalise_text(raw) == expected

    def test_supports_only_configured_codes(self, monkeypatch):
        from tts_service.engines import pokot_vits_engine as mod

        monkeypatch.setattr(
            mod, "POKOT_MODELS", {"pko-p1": {"repo": "o/r", "speaker": None}}
        )
        engine = mod.PokotVitsEngine()
        assert engine.supports_language("pko-p1")
        assert not engine.supports_language("pko-p3")

    @pytest.mark.asyncio
    async def test_rejects_unconfigured_language(self, monkeypatch):
        from tts_service.engines import pokot_vits_engine as mod

        monkeypatch.setattr(mod, "POKOT_MODELS", {})
        with pytest.raises(ValueError):
            await mod.PokotVitsEngine().synthesize("poru rop", "pko-p1")

    @pytest.mark.asyncio
    async def test_rejects_text_without_pokot_letters(self, monkeypatch):
        from tts_service.engines import pokot_vits_engine as mod

        monkeypatch.setattr(
            mod, "POKOT_MODELS", {"pko-p1": {"repo": "o/r", "speaker": None}}
        )
        with pytest.raises(ValueError):
            await mod.PokotVitsEngine().synthesize("123 !?", "pko-p1")

    @pytest.mark.asyncio
    async def test_synthesize_with_stub_model(self, monkeypatch):
        """Full path with a stub model: speaker choice, speed and WAV encoding."""
        import base64

        import numpy as np
        import torch

        from tts_service.engines import pokot_vits_engine as mod

        monkeypatch.setattr(
            mod, "POKOT_MODELS", {"pko-p3": {"repo": "o/r", "speaker": "PK03"}}
        )
        seen = {}

        class StubTokenizer:
            def text_to_ids(self, text):
                seen["text"] = text
                return [1, 2, 3]

        class StubModel:
            speaker_ids = {"PK01": 0, "PK02": 1, "PK03": 2}
            tokenizer = StubTokenizer()
            config = type(
                "C",
                (),
                {"model_args": type("A", (), {"use_speaker_embedding": True})()},
            )()

            def inference(self, x, aux_input):
                seen["speaker_ids"] = aux_input["speaker_ids"].tolist()
                seen["length_scale"] = self.length_scale
                return {"model_outputs": torch.zeros(1, 1, 1600)}

        engine = mod.PokotVitsEngine()
        engine._models["pko-p3"] = StubModel()

        result = await engine.synthesize("Ngala achey", "pko-p3", speed=2.0)

        assert result["engine"] == "pokot_vits"
        assert result["format"] == "wav"
        assert result["sample_rate"] == 16000
        wav = base64.b64decode(result["audio_base64"])
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
        assert seen["text"] == "ŋala acey"
        assert seen["speaker_ids"] == [2]  # configured speaker "PK03"
        assert seen["length_scale"] == pytest.approx(0.5)
        assert np.frombuffer(wav[44:], dtype=np.int16).shape[0] == 1600

    @pytest.mark.asyncio
    async def test_unknown_speaker_is_rejected(self, monkeypatch):
        from tts_service.engines import pokot_vits_engine as mod

        monkeypatch.setattr(
            mod, "POKOT_MODELS", {"pko-p3": {"repo": "o/r", "speaker": "XX"}}
        )

        class StubModel:
            speaker_ids = {"PK01": 0}
            tokenizer = type("T", (), {"text_to_ids": lambda self, t: [1]})()
            config = type(
                "C",
                (),
                {"model_args": type("A", (), {"use_speaker_embedding": True})()},
            )()

        engine = mod.PokotVitsEngine()
        engine._models["pko-p3"] = StubModel()
        with pytest.raises(ValueError, match="Unknown speaker"):
            await engine.synthesize("poru rop", "pko-p3")


class TestPokotLanguages:
    def test_pokot_languages_listed_with_engine(self):
        langs = client.get("/languages").json()
        for code in ("pko-p1", "pko-p2", "pko-p3"):
            assert code in langs
            assert langs[code]["preferred_engine"] == "pokot_vits"
        assert langs["pko"]["preferred_engine"] != "pokot_vits"


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


class TestTurkanaVitsEngine:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Tana Joko", "tana joko"),
            ("ngibaren  kalo\n wukor", "ŋibaren kalo wukor"),
            ("shu thaa chal", "ʃu θaa cal"),
            ("nyakuj, ekile!", "nyakuj ekile"),
            ("a-b d’e", "a-b d'e"),
            ("ŋibaren", "ŋibaren"),
            ("(tana) joko?", "tana joko"),
            ("?! ...", ""),
        ],
    )
    def test_normalise_text(self, raw, expected):
        from tts_service.engines.turkana_vits_engine import normalise_text

        assert normalise_text(raw) == expected

    def test_normalise_text_rejects_digits(self):
        from tts_service.engines.turkana_vits_engine import normalise_text

        with pytest.raises(ValueError, match="digits"):
            normalise_text("ekile 12")

    def test_supports_only_configured_codes(self, monkeypatch):
        from tts_service.engines import turkana_vits_engine as mod

        monkeypatch.setattr(
            mod, "TURKANA_MODELS", {"tuv-open": {"repo": "o/r", "speaker": None}}
        )
        engine = mod.TurkanaVitsEngine()
        assert engine.supports_language("tuv-open")
        assert not engine.supports_language("tuv-col")

    @pytest.mark.asyncio
    async def test_rejects_unconfigured_language(self, monkeypatch):
        from tts_service.engines import turkana_vits_engine as mod

        monkeypatch.setattr(mod, "TURKANA_MODELS", {})
        with pytest.raises(ValueError):
            await mod.TurkanaVitsEngine().synthesize("tana joko", "tuv-open")

    @pytest.mark.asyncio
    async def test_rejects_text_without_turkana_letters(self, monkeypatch):
        from tts_service.engines import turkana_vits_engine as mod

        monkeypatch.setattr(
            mod, "TURKANA_MODELS", {"tuv-open": {"repo": "o/r", "speaker": None}}
        )
        with pytest.raises(ValueError):
            await mod.TurkanaVitsEngine().synthesize("?! ...", "tuv-open")

    @pytest.mark.asyncio
    async def test_synthesize_with_stub_model(self, monkeypatch):
        """Full path with a stub model: speaker choice, speed and WAV encoding."""
        import base64

        import numpy as np
        import torch

        from tts_service.engines import turkana_vits_engine as mod

        monkeypatch.setattr(
            mod, "TURKANA_MODELS", {"tuv-col": {"repo": "o/r", "speaker": "T05"}}
        )
        seen = {}

        class StubTokenizer:
            def text_to_ids(self, text):
                seen["text"] = text
                return [1, 2, 3]

        class StubModel:
            speaker_ids = {"T01": 0, "T02": 1, "T05": 2}
            tokenizer = StubTokenizer()
            config = type(
                "C",
                (),
                {"model_args": type("A", (), {"use_speaker_embedding": True})()},
            )()

            def inference(self, x, aux_input):
                seen["speaker_ids"] = aux_input["speaker_ids"].tolist()
                seen["length_scale"] = self.length_scale
                return {"model_outputs": torch.zeros(1, 1, 1600)}

        engine = mod.TurkanaVitsEngine()
        engine._models["tuv-col"] = StubModel()

        result = await engine.synthesize("Ngibaren shu", "tuv-col", speed=2.0)

        assert result["engine"] == "turkana_vits"
        assert result["format"] == "wav"
        assert result["sample_rate"] == 16000
        wav = base64.b64decode(result["audio_base64"])
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
        assert seen["text"] == "ŋibaren ʃu"
        assert seen["speaker_ids"] == [2]  # configured speaker "T05"
        assert seen["length_scale"] == pytest.approx(0.5)
        assert np.frombuffer(wav[44:], dtype=np.int16).shape[0] == 1600

    @pytest.mark.asyncio
    async def test_first_voice_is_the_default(self, monkeypatch):
        from tts_service.engines import turkana_vits_engine as mod

        monkeypatch.setattr(
            mod, "TURKANA_MODELS", {"tuv-open": {"repo": "o/r", "speaker": None}}
        )
        seen = {}

        class StubModel:
            speaker_ids = {"GRN": 0}
            tokenizer = type("T", (), {"text_to_ids": lambda self, t: [1]})()
            config = type(
                "C",
                (),
                {"model_args": type("A", (), {"use_speaker_embedding": True})()},
            )()

            def inference(self, x, aux_input):
                import torch

                seen["speaker_ids"] = aux_input["speaker_ids"].tolist()
                return {"model_outputs": torch.zeros(1, 1, 160)}

        engine = mod.TurkanaVitsEngine()
        engine._models["tuv-open"] = StubModel()
        await engine.synthesize("tana joko", "tuv-open")
        assert seen["speaker_ids"] == [0]

    @pytest.mark.asyncio
    async def test_unknown_speaker_is_rejected(self, monkeypatch):
        from tts_service.engines import turkana_vits_engine as mod

        monkeypatch.setattr(
            mod, "TURKANA_MODELS", {"tuv-col": {"repo": "o/r", "speaker": "XX"}}
        )

        class StubModel:
            speaker_ids = {"T01": 0}
            tokenizer = type("T", (), {"text_to_ids": lambda self, t: [1]})()
            config = type(
                "C",
                (),
                {"model_args": type("A", (), {"use_speaker_embedding": True})()},
            )()

        engine = mod.TurkanaVitsEngine()
        engine._models["tuv-col"] = StubModel()
        with pytest.raises(ValueError, match="Unknown speaker"):
            await engine.synthesize("tana joko", "tuv-col")


class TestTurkanaLanguages:
    def test_turkana_languages_listed_with_engine(self):
        langs = client.get("/languages").json()
        for code in ("tuv-open", "tuv-col"):
            assert code in langs
            assert langs[code]["preferred_engine"] == "turkana_vits"
        assert langs["tuv"]["preferred_engine"] != "turkana_vits"
