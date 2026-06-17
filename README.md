# Husika TTS

**The voice layer of the Husika early-warning platform** — multilingual
Text-to-Speech that turns written drought and weather alerts into spoken audio
in the languages of the Greater Horn of Africa.

> **Developed by [Speedykom GmbH](https://speedykom.de)** for the Husika
> early-warning platform operated by **ICPAC** (IGAD Climate Prediction and
> Applications Centre). Produced under the **Peaceful and Resilient Borderlands
> Programme (PRBP)**, implemented by **GIZ** within the **SCIDA III** framework in
> support of **IGAD**, and **co-funded by the European Union and the German Federal
> Ministry for Economic Cooperation and Development (BMZ)**.

<p align="center">
  <img src="assets/logos/eu.png" alt="Co-funded by the European Union" height="52">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/german-cooperation-bmz.png" alt="German Cooperation — BMZ" height="52">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/giz.png" alt="Implemented by GIZ" height="52">
</p>

<p align="center">
  <img src="assets/logos/igad.png" alt="IGAD" height="52">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/icpac.png" alt="ICPAC" height="52">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/speedykom.png" alt="Developed by Speedykom" height="52">
</p>

<p align="center"><sub>
  <b>Co-funded by the European Union</b> and the German Federal Ministry for
  Economic Cooperation and Development (BMZ) &nbsp;·&nbsp; <b>Implemented by GIZ</b>
  &nbsp;·&nbsp; In partnership with IGAD &amp; ICPAC &nbsp;·&nbsp; Developed by Speedykom
</sub></p>

---

## What is Husika?

*Husika* is a Swahili word meaning **"to be concerned, to be involved"** — to be
reached, and to take part. The Husika platform ([husika.icpac.net](https://husika.icpac.net))
is an early-warning information system operated by ICPAC that delivers weather
forecasts, drought updates, crop advisories and early-warning alerts to
smallholder farmers and pastoralists across the IGAD region over web, mobile
and SMS.

This repository is the **Text-to-Speech (TTS) service** behind that platform.
It converts the alerts Husika sends into natural-sounding speech, so that people
who cannot read a text message — or who speak a language that is rarely written —
can still **hear** the warning in their own language and act on it. Send it
**text + a language code**, and it returns **spoken audio**.

## Why it matters

Drought in the Greater Horn of Africa affects **over 18 million** smallholder
farmers and pastoralists. Cross-border communities in the **Karamoja Cluster**
(northern Kenya and north-eastern Uganda) are among the hardest hit. A warning
only helps if it is understood — yet many of the people most at risk speak
languages with little or no digital voice technology, and may not read at all.

Husika TTS closes that **"last mile"**: it speaks the alert out loud, in the
listener's own language, on a basic phone or over a loudspeaker.

## Languages

The service spans regional lingua francas and low-resource cross-border
languages alike:

- **Wide-reach languages** — Swahili (~100M speakers), Oromo (~45M),
  Amharic (~32M), Somali (~20M), plus Arabic, English and French.
- **National & regional languages** — Tigrinya, Luganda, Kinyarwanda, Kirundi,
  Nuer, Dinka.
- **Low-resource Karamoja Cluster languages** — Turkana, Karamojong and Pokot:
  pastoralist languages with little prior speech technology, served by custom
  fine-tuned voice models built for this project.

See [Language Support](#language-support) below for the full matrix.

## Innovation

- **Voice for languages that had none.** For several Karamoja Cluster languages
  this is among the first working text-to-speech ever built — purpose-trained so
  warnings reach non-literate, cross-border pastoralist communities.
- **Right engine per language.** A router automatically picks the best available
  voice for each language and falls back gracefully, so coverage degrades
  gradually instead of failing.
- **Works at the edge.** A fast, offline, GPU-free path keeps the system usable
  in low-connectivity field conditions, with higher-quality neural voices where
  resources allow.

---

## Architecture

```
 Text + Language Code
         │
         ▼
   ┌───────────┐
   │  FastAPI   │  POST /tts
   │  Server    │  GET  /languages
   └─────┬─────┘
         │
   ┌─────▼──────┐
   │   Engine    │  picks engine based on languages.yaml
   │   Router    │  fallback: preferred → mms → espeak
   └──┬──────┬──┘
      │      │
 ┌────▼──┐ ┌─▼────────┐
 │eSpeak │ │ Meta MMS  │
 │  NG   │ │(HuggingFace)│
 └───────┘ └───────────┘
      │      │
      ▼      ▼
   WAV audio (base64)
         │
   ┌─────▼─────┐
   │  Web UI    │  text input, language picker, audio playback
   └───────────┘
```

**Engines:**

| Engine | How it works | Strengths |
|--------|-------------|-----------|
| **eSpeak NG** | Rule-based, runs locally via CLI | Fast (<100 ms), offline, no GPU |
| **Meta MMS** | Neural model from HuggingFace (`facebook/mms-tts-*`) | 1200+ languages, better quality |
| **Custom VITS** | In-house fine-tuned neural voices | Low-resource languages with no off-the-shelf voice |

---

## Folder Structure

```
husika-tts/
├── tts_service/
│   ├── api/
│   │   ├── server.py          # FastAPI app, endpoints
│   │   └── schemas.py         # Request/response models
│   └── engines/
│       ├── base.py            # Abstract engine interface
│       ├── espeak_engine.py   # eSpeak NG (subprocess)
│       ├── mms_engine.py      # Meta MMS (HuggingFace)
│       └── engine_router.py   # Language → engine routing
├── web-ui/
│   └── index.html             # Browser test UI
├── languages/
│   └── languages.yaml         # Language config (single source of truth)
├── tests/
│   └── test_tts_service.py
├── pyproject.toml
├── Makefile
├── Dockerfile
└── .env.example
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **espeak-ng** installed (`brew install espeak` / `apt install espeak-ng`)
- **[ruff](https://docs.astral.sh/ruff/)** — linter and formatter (installed with dev dependencies)

### Setup

```bash
make install        # uv sync
make dev-install    # uv sync --extra dev (includes test & lint tools)
```

### Run

```bash
make run
# runs: uv run uvicorn tts_service.api.server:app --host 0.0.0.0 --port 8181 --reload
```

- **API docs:** http://localhost:8181/docs
- **Test UI:** http://localhost:8181/ui
- **Health:** http://localhost:8181/health

### Test & Lint

```bash
make test           # uv run pytest tests/ -v
make lint           # uv run ruff check tts_service/ tests/
make format         # uv run ruff format tts_service/ tests/
make clean          # remove caches and build artifacts
```

---

## Run with Docker

No local Python setup required — just Docker.

```bash
docker build -t husika-tts .
docker run -p 8181:8000 husika-tts
```

Then open http://localhost:8181/ui in your browser.

First synthesis per language downloads the MMS model from HuggingFace (~150 MB each), so the initial request is slow. Subsequent requests are fast.

---

## API

### `POST /tts`

```json
{
  "text": "Habari, hii ni ujumbe wa majaribio.",
  "lang_code": "swa",
  "speed": 1.0
}
```

Response:

```json
{
  "audio_base64": "UklGR...",
  "format": "wav",
  "sample_rate": 22050,
  "lang_code": "swa",
  "engine": "mms"
}
```

### `GET /languages`

Returns all configured languages with their engine availability.

### `GET /health`

Returns `{"status": "ok"}`.

---

## Language Support

| Language | Code | eSpeak NG | MMS | Custom VITS | Preferred |
|----------|------|-----------|-----|-------------|-----------|
| Swahili | `swa` | yes | yes | — | mms |
| Amharic | `amh` | — | yes | — | mms |
| Arabic | `ara` | yes | yes | — | mms |
| Somali | `som` | — | yes | — | mms |
| Oromo | `orm` | — | yes | — | mms |
| Tigrinya | `tir` | yes | yes | — | mms |
| Luganda | `lug` | — | yes | — | mms |
| Kinyarwanda | `kin` | — | yes | — | mms |
| Kirundi | `rn` | — | yes | — | mms |
| Nuer | `nue` | — | yes | — | mms |
| Dinka | `din` | — | yes | — | mms |
| Turkana | `tuv` | — | — | yes | custom |
| Karamojong | `kdj` | — | — | yes | custom |
| Pokot | `pko` | — | yes | — | mms |
| English | `en` | yes | yes | — | espeak |
| French | `fr` | yes | yes | — | espeak |

To add a language: add an entry to `languages/languages.yaml` and ensure an engine supports it (add a mapping in the relevant engine).

---

## Adding a New Engine

1. Create a class in `tts_service/engines/` that extends `TTSEngine` from `base.py`
2. Implement `synthesize()`, `supports_language()`, and the `name` property
3. Register it in `EngineRouter.__init__`

---

## Credits & License

**Developed by Speedykom GmbH** for the Husika early-warning platform operated by
**ICPAC** (IGAD Climate Prediction and Applications Centre). Produced under the
**Peaceful and Resilient Borderlands Programme (PRBP)**, implemented by **GIZ**
within the **SCIDA III** framework in support of **IGAD**, and **co-funded by the
European Union and the German Federal Ministry for Economic Cooperation and
Development (BMZ)**.

- **Backend code (this repository):** Copyright (c) 2026 Speedykom GmbH and
  Deutsche Gesellschaft für Internationale Zusammenarbeit (GIZ) GmbH. Licensed
  under the Apache License 2.0 — see [`LICENSE`](LICENSE). Attribution that must
  be preserved in derivative works is listed in [`NOTICE`](NOTICE).

- **Mobile app:** The mobile application lives in a separate repository
  (`github.com/Speedykom/husika-tts-mobile`) and is licensed under MIT;
  see that repository's `LICENSE` file.

> **Disclaimer.** Views and opinions expressed are those of the authors only and
> do not necessarily reflect those of the European Union, BMZ, GIZ, IGAD or ICPAC.
> Neither the European Union nor any other funding or implementing party can be
> held responsible for them.
