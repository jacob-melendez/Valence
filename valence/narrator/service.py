"""Narrator HTTP service (local only, 127.0.0.1:8081).

``POST /narrate`` takes one fact sheet (validated with :class:`FactSheet`; 422 on bad input) and
returns the explanation, its source and latency; ``GET /health`` reports the model, warm-up state
and per-source counts. The model is loaded and warmed once at startup; if that fails the service
keeps answering with templates and says so in ``/health``.

    python -m valence.narrator.service [--port 8081] [--template-only]
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from valence.narrator.engine import Narrator, TemplateOnlyNarrator
from valence.narrator.schema import FactSheet

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 8081


class NarratorLike(Protocol):
    def load(self) -> None: ...
    def narrate(self, sheet: dict[str, Any]) -> dict[str, Any]: ...
    def health(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class NarrateResponse(BaseModel):
    text: str
    source: Literal["llm", "llm_retry", "template"]
    latency_ms: float
    attempts: int


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

    @app.post("/narrate", response_model=NarrateResponse)
    def narrate(sheet: FactSheet) -> NarrateResponse:
        result = narrator.narrate(sheet.model_dump(mode="json"))
        return NarrateResponse(
            text=result["text"],
            source=result["source"],
            latency_ms=result["latency_ms"],
            attempts=result["attempts"],
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return narrator.health()

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Valence narrator service.")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--template-only", action="store_true", help="never load the LLM")
    parser.add_argument("--model", default=None, help="local mlx model dir (default: VALENCE_NARRATOR_MODEL)")
    parser.add_argument("--adapter", default=None, help="local LoRA adapter dir (default: VALENCE_NARRATOR_ADAPTER)")
    args = parser.parse_args(argv)

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    narrator = TemplateOnlyNarrator() if args.template_only else Narrator(args.model, args.adapter)
    uvicorn.run(create_app(narrator), host=HOST, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
