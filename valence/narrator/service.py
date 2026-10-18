"""Narrator HTTP service (local only, 127.0.0.1:8081), contract 1.0.

``POST /narrate`` takes one ``valence.contracts.FactSheet`` (FastAPI validates it with the
contract model itself, so anything off-contract is a 422) and returns exactly a
``valence.contracts.NarrationResult`` ``{text, source, latency_ms, attempts}``, which is what
``valence.explain.narrator_client.NarratorClient`` parses. ``GET /health`` is 200 only when the
model is loaded and warmed up (docs/API.md 9.2; the client only checks the status code) and 503
otherwise, with the model, warm-up state, load error and per-source counts in the body either way.
The model is loaded, its prompt caches prefilled and warmed once at startup; if that fails the
service keeps answering ``/narrate`` with templates and ``/health`` says 503.

    python -m valence.narrator.service [--port 8081] [--template-only] [--adapter PATH | --no-adapter] \\
        [--model PATH] [--prompt-style few_shot|compact]

Which arm runs is the A/B ship decision (``engine.SHIP``, overridable with ``VALENCE_NARRATOR_SHIP=lora|prompt``):
with "lora" and ``adapters/narrator`` present the service loads that adapter, otherwise the base model.
``--adapter`` / ``VALENCE_NARRATOR_ADAPTER`` override it, and ``--no-adapter`` (or the env value
"none") forces prompt-only.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from valence.contracts import FactSheet, NarrationResult
from valence.narrator.engine import Narrator, TemplateOnlyNarrator, ship_choice, startup_adapter
from valence.narrator.prompt import PROMPT_STYLES

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 8081


class NarratorLike(Protocol):
    def load(self) -> None: ...
    def narrate(self, sheet: dict[str, Any]) -> dict[str, Any]: ...
    def health(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


def create_app(narrator: NarratorLike | None = None, *, template_only: bool = False) -> FastAPI:
    """App factory; tests inject a fake narrator, production builds the MLX one."""
    narrator = narrator or (TemplateOnlyNarrator() if template_only else Narrator())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await run_in_threadpool(narrator.load)
        except Exception as exc:
            logger.warning("narrator model unavailable (%s); serving template explanations", type(exc).__name__)
        try:
            yield
        finally:
            narrator.close()

    app = FastAPI(title="Valence Narrator", version="1.0", lifespan=lifespan)
    app.state.narrator = narrator

    @app.post("/narrate", response_model=NarrationResult)
    def narrate(sheet: FactSheet) -> NarrationResult:
        out = narrator.narrate(sheet.model_dump(mode="json"))
        return NarrationResult(text=out["text"], source=out["source"], latency_ms=int(round(out["latency_ms"])),
                               attempts=int(out.get("attempts", 1)))

    @app.get("/health")
    def health() -> JSONResponse:
        """200 only when the LLM is loaded and warmed up; 503 (same body) while loading, after a
        failed load and in template-only mode. The backend checks only the status code."""
        body = narrator.health()
        return JSONResponse(status_code=200 if body.get("ok") else 503, content=body)

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Valence narrator service.")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--template-only", action="store_true", help="never load the LLM")
    parser.add_argument("--model", default=None, help="local mlx model dir (default: VALENCE_NARRATOR_MODEL)")
    parser.add_argument("--adapter", default=None,
                        help="local LoRA adapter dir (default: VALENCE_NARRATOR_ADAPTER, else the shipped choice)")
    parser.add_argument("--no-adapter", action="store_true", help="serve the base model (prompt-only)")
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default=None,
                        help="default: compact with an adapter, few_shot without")
    args = parser.parse_args(argv)

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.template_only:
        narrator: Narrator = TemplateOnlyNarrator()
    else:
        adapter = None if args.no_adapter else startup_adapter(args.adapter)
        narrator = Narrator(args.model, adapter or "none", prompt_style=args.prompt_style)
        logger.info("narrator arm: %s (ship=%s, adapter=%s, prompt_style=%s)", "lora" if adapter else "prompt",
                    ship_choice(), adapter, narrator.prompt_style)
    uvicorn.run(create_app(narrator), host=HOST, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
