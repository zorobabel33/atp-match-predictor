"""Per-process JSON telemetry with bounded labels and a 500-prediction drift window."""

import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import deque

import pandas as pd
import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.responses import JSONResponse

from ml.contract import FEATURES
from ml.model import compare_distributions
from ml.policy import DRIFT_THRESHOLD
from ml.registry import configure_tracking, load_training_profile


class _Stdout:
    """Follow stdout if the host redirects it after application construction."""

    def write(self, value):
        return sys.stdout.write(value)

    def flush(self):
        return sys.stdout.flush()


def make_logger():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    return structlog.wrap_logger(
        structlog.WriteLogger(file=_Stdout()),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)),
        processors=[structlog.contextvars.merge_contextvars, structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso", utc=True),
                    structlog.processors.JSONRenderer()],
    )


class Telemetry:
    def __init__(self):
        self.log = make_logger()
        self.registry = CollectorRegistry()
        self.requests = Counter("http_requests_total", "HTTP requests",
                                ["method", "path", "status"], registry=self.registry)
        self.duration = Histogram("http_request_duration_seconds", "HTTP request duration",
                                  ["path"], registry=self.registry)
        self.predictions = Counter("predictions_total", "Successful predictions",
                                   ["prediction", "model_version"], registry=self.registry)
        self.model = Gauge("model_info", "Loaded approved model", ["model_version"],
                           registry=self.registry)
        self.drift = Gauge("feature_drift", "Population Stability Index", ["feature"],
                           registry=self.registry)
        self.reference_ready = Gauge("drift_reference_ready", "Matching training profile loaded",
                                     registry=self.registry)
        self.window_size = Gauge("drift_window_size", "Predictions in the drift window",
                                 registry=self.registry)
        Gauge("feature_drift_threshold", "Investigate drift at this PSI", registry=self.registry
              ).set(DRIFT_THRESHOLD)
        for status in ("200", "422", "500", "503"):
            self.requests.labels("POST", "/predict", status)
        self.reference = None
        self.window = deque(maxlen=500)
        self.count = 0
        self.lock = threading.Lock()

    def load_reference(self, info):
        try:
            client, _ = configure_tracking()
            version = client.get_model_version(info["model_name"], info["model_version"])
            reference = load_training_profile(client, version)
        except Exception as exc:
            self.log.warning("drift_reference_unavailable", model_version=info["model_version"],
                             error_type=type(exc).__name__)
            return
        with self.lock:
            self.reference = reference
            self.reference_ready.set(1)
        self.log.info("drift_reference_loaded", model_version=info["model_version"],
                      dataset_version=reference["dataset_version"])

    def prediction(self, payload, score, version, latency):
        prediction = "1" if score >= 0.5 else "0"
        self.predictions.labels(prediction, version).inc()
        self.log.info("prediction_made", model_version=version, prediction=prediction, score=score,
                      latency_ms=round(latency * 1000, 3), number_of_features=len(FEATURES),
                      number_of_missing_values=0)
        with self.lock:
            self.window.append({name: payload[name] for name in FEATURES})
            self.count += 1
            self.window_size.set(len(self.window))
            if self.reference is None or len(self.window) < 500 or self.count % 100:
                return
            drift = compare_distributions(self.reference, pd.DataFrame(self.window))
            for name, value in drift.items():
                self.drift.labels(name).set(value)
        self.log.info("drift_calculated", model_version=version, window_size=500,
                      drifted_features=[n for n, v in drift.items() if v >= DRIFT_THRESHOLD])


class RequestTelemetry:
    def __init__(self, app, telemetry):
        self.app, self.telemetry = app, telemetry

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        incoming = headers.get(b"x-request-id", b"").decode("ascii", errors="replace")
        request_id = (incoming if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", incoming)
                      else uuid.uuid4().hex)
        tokens = structlog.contextvars.bind_contextvars(request_id=request_id)
        started, status, sent = time.perf_counter(), 500, False
        error_type = None

        async def observed_send(message):
            nonlocal status, sent
            if message["type"] == "http.response.start":
                status, sent = message["status"], True
                message["headers"] = [(k, v) for k, v in message.get("headers", [])
                                      if k.lower() != b"x-request-id"]
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        except Exception as exc:
            error_type = type(exc).__name__
            if sent:
                raise
            await JSONResponse({"detail": "Internal server error"}, status_code=500)(
                scope, receive, observed_send)
        finally:
            elapsed = time.perf_counter() - started
            route = scope.get("route")
            # Never label or log arbitrary paths, query strings, request bodies or headers.
            path = getattr(route, "path", None) or "/unmatched"
            if scope.get("path", "").startswith("/static/"):
                path = "/static/*"
            method = scope.get("method", "OTHER")
            if method not in {"GET", "POST", "HEAD", "PUT", "PATCH", "DELETE", "OPTIONS"}:
                method = "OTHER"
            self.telemetry.requests.labels(method, path, str(status)).inc()
            self.telemetry.duration.labels(path).observe(elapsed)
            log = (self.telemetry.log.error if status >= 400 or error_type
                   else self.telemetry.log.info)
            log("request_finished", method=method, path=path, status=status,
                duration_ms=round(elapsed * 1000, 3), **({"error_type": error_type}
                                                     if error_type else {}))
            structlog.contextvars.reset_contextvars(**tokens)
