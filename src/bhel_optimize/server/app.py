"""Flask UI for the demo: boiler diagram, live panel, graphs, scenario control.

Run it with::

    uv run python -m bhel_optimize.server            # http://127.0.0.1:5000
    uv run python -m bhel_optimize.server --speed 20 # fast-forward playback

The UI is server-rendered Jinja plus a small vanilla-JS poller; there is no SPA
framework (project rule). It reuses the simulation layer for playback/fault
injection and the Overview Agent for the situation panel.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request, url_for

from bhel_optimize import config as C
from bhel_optimize.jsonio import json_safe
from bhel_optimize.overview.agent import OverviewAgent
from bhel_optimize.overview.api import situation_payload
from bhel_optimize.simulation import faults as F

LEVEL_COLOURS = {
    "NOMINAL": "#1f9d55",
    "WATCH": "#d9a406",
    "ALERT": "#e2701a",
    "CRITICAL": "#c62828",
}


def live_payload(agent: OverviewAgent, controller, *, advance: bool = True) -> dict:
    """Trimmed payload for fast polling: levels, scores and current values."""
    if advance:
        controller.tick()
    overview = agent.update(
        controller.frame, controller.position, context=controller.snapshot()["labels"]
    )
    snapshot = controller.snapshot()
    payload = {
        "position": overview["position"],
        "timestamp": overview["timestamp"],
        "total_samples": snapshot["total_samples"],
        "at_end": snapshot["at_end"],
        "mode": snapshot["mode"],
        "speed": snapshot["speed"],
        "scenario": snapshot["scenario"],
        "labels": snapshot["labels"],
        "situation": {
            "level": overview["situation"]["level"],
            "counts": overview["situation"]["counts"],
            "summary": overview["situation"]["summary"],
            "narrative_source": overview["situation"].get("narrative_source"),
            "narrative_model": overview["situation"].get("narrative_model"),
            "criticality_escalated": overview["situation"]["criticality_escalated"],
        },
        "parts": {
            key: {
                "name": info["name"],
                "level": info["level"],
                "score": info["score"],
                "threshold": info["threshold"],
                "trend": info["trend"],
                "severity_iqr": info["severity_iqr"],
                "values": info["values"],
                "deviations": info["deviations"],
                "top_tags": info["top_tags"],
                "pattern": info["pattern"],
            }
            for key, info in overview["parts"].items()
        },
    }
    if payload["scenario"]:
        duration = snapshot["scenario"].get("duration_s")
        payload["scenario"]["held"] = duration is None or duration == float("inf")
    return json_safe(payload)


def scenario_state(controller) -> dict:
    """Current playback/injection state for the control panel."""
    status = controller.status()
    state = {
        "mode": status["mode"],
        "position": status["position"],
        "total_samples": status["total_samples"],
        "speed": status["speed"],
        "scenario_key": status["scenario_key"],
        "fault_key": status["fault_key"],
        "fault_part": status["fault_part"],
        "fault_part_name": (
            C.PARTS[status["fault_part"]].name if status["fault_part"] else None
        ),
        "elapsed_s": status["elapsed_s"],
    }
    active = controller.active
    state["held"] = bool(active and active.duration_s == float("inf"))
    state["duration_s"] = None if not active or active.duration_s == float("inf") else active.duration_s
    return json_safe(state)


def create_app(
    agent: OverviewAgent | None = None,
    controller=None,
    *,
    speed: float = 5.0,
    name: str = "bhel_optimize",
) -> Flask:
    """Build the Flask app. Without arguments, real artifacts are loaded."""
    if controller is None:
        from bhel_optimize.simulation import ScenarioController

        controller = ScenarioController(mode="livestream", speed=speed)
    if agent is None:
        agent = OverviewAgent.from_artifacts(controller.frame)

    app = Flask(
        name,
        template_folder=str(C.PROJECT_ROOT / "templates"),
        static_folder=str(C.PROJECT_ROOT / "static"),
    )
    # Static assets are cached aggressively by default, which served a stale
    # app.css against fresh markup (the schematic rendered as black boxes).
    # Cache-bust by mtime and let the browser revalidate every time.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    # see jsonio: non-finite floats are invalid JSON and break browsers
    app.json.allow_nan = False
    app.config["AGENT"] = agent
    app.config["CONTROLLER"] = controller

    @app.template_global()
    def static_url(filename: str) -> str:
        """``url_for('static', ...)`` with a content timestamp query."""
        path = C.PROJECT_ROOT / "static" / filename
        try:
            stamp = int(path.stat().st_mtime)
        except OSError:
            stamp = 0
        return url_for("static", filename=filename, v=stamp)

    # -- pages ---------------------------------------------------------
    @app.get("/")
    def boiler_page():
        return render_template(
            "boiler.html",
            parts=C.PARTS,
            tag_info=C.TAG_INFO,
            part_keys=C.part_keys(),
            scenarios=F.list_scenarios(),
            level_colours=LEVEL_COLOURS,
            interval_s=C.SAMPLE_INTERVAL_S,
        )

    @app.get("/part/<part_key>")
    def part_page(part_key: str):
        if part_key not in C.PARTS:
            return render_template("not_found.html", part_key=part_key), 404
        ensemble = agent.ensembles.get(part_key)
        return render_template(
            "part.html",
            part=C.PARTS[part_key],
            part_key=part_key,
            ensemble=ensemble,
            tags=[(tag, C.TAG_INFO[tag]) for tag in C.PARTS[part_key].tags],
            band=agent.bands.get(part_key),
            level_colours=LEVEL_COLOURS,
        )

    # -- data APIs -----------------------------------------------------
    @app.get("/api/live")
    def api_live():
        return jsonify(live_payload(agent, controller)), 200

    @app.get("/api/situation")
    def api_situation():
        return jsonify(situation_payload(agent, controller, advance=True)), 200

    @app.get("/api/scenarios")
    def api_scenarios():
        return (
            jsonify(
                json_safe({
                    "scenarios": F.list_scenarios(),
                    "state": scenario_state(controller),
                    "parts": {
                        key: {
                            "name": part.name,
                            "criticality": part.criticality,
                            "tags": list(part.tags),
                        }
                        for key, part in C.PARTS.items()
                    },
                    "interval_s": C.SAMPLE_INTERVAL_S,
                })
            ),
            200,
        )

    @app.post("/api/scenarios/start")
    def api_start():
        body = request.get_json(silent=True) or {}
        key = body.get("scenario_key") or request.args.get("scenario_key")
        if key not in F.SCENARIOS:
            return jsonify({"error": f"unknown scenario: {key}"}), 400
        controller.start_scenario(
            key,
            duration_s=body.get("duration_s"),
            intensity=float(body.get("intensity", 1.0)),
            target_tag=body.get("target_tag"),
            # the UI holds a fault until Stop so the demo does not end in
            # seconds when playback runs fast
            hold=bool(body.get("hold", True)),
        )
        return jsonify({"state": scenario_state(controller), "live": live_payload(agent, controller)}), 200

    @app.post("/api/scenarios/stop")
    def api_stop():
        controller.stop_scenario()
        return jsonify({"state": scenario_state(controller)}), 200

    @app.post("/api/mode")
    def api_mode():
        body = request.get_json(silent=True) or {}
        mode = body.get("mode") or request.args.get("mode")
        if mode not in ("livestream", "simulation"):
            return jsonify({"error": f"unknown mode: {mode}"}), 400
        controller.set_mode(mode)
        agent.reset()
        return jsonify({"state": scenario_state(controller)}), 200

    @app.post("/api/speed")
    def api_speed():
        body = request.get_json(silent=True) or {}
        try:
            speed_value = float(body.get("speed", request.args.get("speed", controller.speed)))
        except (TypeError, ValueError):
            return jsonify({"error": "speed must be numeric"}), 400
        if speed_value <= 0:
            return jsonify({"error": "speed must be > 0"}), 400
        controller.set_speed(speed_value)
        return jsonify({"state": scenario_state(controller)}), 200

    @app.post("/api/reset")
    def api_reset():
        body = request.get_json(silent=True) or {}
        start = int(body.get("start_index", 0))
        controller.reset(start)
        agent.reset()
        return jsonify({"state": scenario_state(controller)}), 200

    @app.get("/api/health")
    def api_health():
        return (
            jsonify(
                json_safe({
                    "status": "ok",
                    "parts": sorted(agent.ensembles),
                    "bands": {
                        key: {
                            "watch": round(band.watch, 4),
                            "alert": round(band.alert, 4),
                            "critical": round(band.critical, 4),
                        }
                        for key, band in agent.bands.items()
                    },
                    "state": scenario_state(controller),
                })
            ),
            200,
        )

    return app


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI glue
    import argparse

    parser = argparse.ArgumentParser(description="Run the boiler monitoring UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--speed", type=float, default=5.0, help="samples per second")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    app = create_app(speed=args.speed)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
