"""Serve the approved champion and an offline player comparison interface."""

import asyncio
import math
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from app.monitoring import RequestTelemetry, Telemetry
from ml.contract import ALLOWED_VALUES, FEATURES, INTEGER, RANGES
from ml.registry import configure_tracking, get_champion, load_version

ROOT = Path(__file__).resolve().parent


class MatchFields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @field_validator("*", check_fields=False)
    @classmethod
    def allowed_value(cls, value, info):
        allowed = ALLOWED_VALUES.get(info.field_name)
        if allowed is not None and value not in allowed:
            raise ValueError(f"Expected one of {allowed}")
        return value


def match_fields():
    fields = {}
    for name in FEATURES:
        kind = int if name in INTEGER else float if name in RANGES else str
        low, high = RANGES.get(name, (None, None))
        fields[name] = (kind, Field(..., ge=low, le=high))
    return fields


MatchInput = create_model("MatchInput", __base__=MatchFields, **match_fields())


class Prediction(BaseModel):
    probability_player_a_wins: float = Field(ge=0, le=1)
    predicted_winner: Literal["player_a", "player_b"]
    model_version: str


def example_match():
    """Build a valid startup probe from the same contract used for requests."""
    result = {}
    for name in FEATURES:
        if name in ALLOWED_VALUES:
            result[name] = ALLOWED_VALUES[name][0]
        else:
            low, high = RANGES[name]
            result[name] = low if high is None else (low + high) / 2
            if name in INTEGER:
                result[name] = int(result[name])
    return result


def probability(model, payload):
    frame = pd.DataFrame([payload], columns=list(FEATURES))
    result = float(model.predict_proba(frame)[0, 1])
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("The model returned an invalid probability")
    return result


def load_champion():
    client, name = configure_tracking()
    version = get_champion(client, name)
    if version is None:
        raise ValueError("No approved champion is available")
    model = load_version(version)
    probability(model, example_match())
    return model, {
        "model_name": name,
        "model_version": str(version.version),
        "run_id": version.run_id,
        "dataset_version": version.tags.get("dataset_version"),
        "data_source": version.tags.get("data_source", "unknown"),
    }


def create_app():
    telemetry = Telemetry()

    @asynccontextmanager
    async def lifespan(application):
        async def initialize():
            try:
                model, info = await asyncio.to_thread(load_champion)
            except Exception as exc:
                # Transport errors can include credentials or signed URLs: log only the type.
                telemetry.log.error("model_load_failed", error_type=type(exc).__name__)
                return
            application.state.model = model
            application.state.model_info = info
            telemetry.model.labels(info["model_version"]).set(1)
            for prediction in ("0", "1"):
                telemetry.predictions.labels(prediction, info["model_version"])
            telemetry.log.info("model_loaded", **info)
            await asyncio.to_thread(telemetry.load_reference, info)

        task = asyncio.create_task(initialize())
        application.state.load_task = task
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        application.state.model = None
        application.state.model_info = None

    application = FastAPI(title="Courtside · ATP match predictor", lifespan=lifespan)
    application.state.model = None
    application.state.model_info = None
    application.state.telemetry = telemetry
    application.add_middleware(RequestTelemetry, telemetry=telemetry)
    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Avoid echoing raw inputs, including non-finite JSON numbers, into the response.
        errors = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(telemetry.registry),
                        headers={"Content-Type": CONTENT_TYPE_LATEST})

    @application.get("/ready")
    def ready(request: Request):
        loaded = request.app.state.model is not None
        return JSONResponse({"status": "ready" if loaded else "not_ready"},
                            status_code=200 if loaded else 503)

    @application.get("/model-info")
    def model_info(request: Request):
        if request.app.state.model_info is None:
            raise HTTPException(503, "The approved model is not ready")
        return request.app.state.model_info

    @application.get("/contract", include_in_schema=False)
    def contract():
        return {"features": FEATURES, "allowed_values": ALLOWED_VALUES,
                "ranges": RANGES, "integer": INTEGER}

    @application.post("/predict", response_model=Prediction)
    def predict(payload: MatchInput, request: Request):
        model = request.app.state.model
        if model is None:
            raise HTTPException(503, "The approved model is not ready")
        try:
            started = time.perf_counter()
            score = probability(model, payload.model_dump())
        except Exception as exc:
            telemetry.log.error("prediction_failed", error_type=type(exc).__name__)
            raise HTTPException(503, "The model could not serve this prediction") from None
        try:
            telemetry.prediction(payload.model_dump(), score,
                                 request.app.state.model_info["model_version"],
                                 time.perf_counter() - started)
        except Exception as exc:
            # Observability failure must never turn a valid inference into an outage.
            telemetry.log.error("monitoring_failed", error_type=type(exc).__name__)
        return Prediction(probability_player_a_wins=score,
                          predicted_winner="player_a" if score >= 0.5 else "player_b",
                          model_version=request.app.state.model_info["model_version"])

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), access_log=False)
