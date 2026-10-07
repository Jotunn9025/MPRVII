"""Minimal Flask API for the Overview Agent (the UI builds on this in M6).

Routes
------
``GET /api/situation``  full situation overview (plant level, per-part levels,
                        trends, deviating tags, probable pattern, explanations)
``GET /api/health``     liveness probe with artifact/part information

The app is deliberately tiny: M6 adds the boiler diagram, live values,
historical graphs and the livestream/simulation dropdown on top of it.
"""

from __future__ import annotations

from flask import Flask, jsonify

from bhel_optimize.jsonio import json_safe
from bhel_optimize.overview.agent import OverviewAgent


def situation_payload(agent: OverviewAgent, controller, *, advance: bool = False) -> dict:
    """Build the ``/api/situation`` payload from the controller's current sample."""
    if advance:
        controller.tick()
    overview = agent.update(controller.frame, controller.position)
    snapshot = controller.snapshot()
    overview["mode"] = snapshot["mode"]
    overview["scenario"] = snapshot["scenario"]
    overview["labels"] = snapshot["labels"]
    overview["speed"] = snapshot["speed"]
    overview["total_samples"] = snapshot["total_samples"]
    overview["at_end"] = snapshot["at_end"]
    if overview.get("scenario"):
        duration = snapshot["scenario"].get("duration_s")
        overview["scenario"]["held"] = duration is None or duration == float("inf")
    return json_safe(overview)


def create_overview_app(agent: OverviewAgent, controller, *, name: str = "bhel_optimize") -> Flask:
    """Create the Flask app exposing the Overview Agent."""
    app = Flask(name)
    # fail loudly if a non-finite float ever reaches the encoder (json_safe
    # should have removed them; a browser would choke on Infinity/NaN)
    app.json.allow_nan = False
    app.config["AGENT"] = agent
    app.config["CONTROLLER"] = controller

    @app.get("/api/health")
    def health() -> tuple:
        return (
            jsonify(
                {
                    "status": "ok",
                    "parts": sorted(agent.ensembles),
                    "thresholds": {
                        key: round(float(e.threshold), 4)
                        for key, e in agent.ensembles.items()
                    },
                    "position": controller.position,
                    "mode": controller.mode,
                }
            ),
            200,
        )

    @app.get("/api/situation")
    def situation() -> tuple:
        return jsonify(situation_payload(agent, controller, advance=True)), 200

    return app
