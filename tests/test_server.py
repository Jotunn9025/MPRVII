"""Tests for the Flask UI layer: pages, live payload, scenario control."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from bhel_optimize import config as C
from bhel_optimize.data import fit_scalers
from bhel_optimize.detection.ensembles import PartEnsemble
from bhel_optimize.detection.windows import part_tags
from bhel_optimize.overview.agent import OverviewAgent
from bhel_optimize.server.app import create_app, live_payload, scenario_state
from bhel_optimize.simulation import ScenarioController


class StubDetector:
    """Scripted detector so the UI can be tested without trained models."""

    key = "stub"
    name = "Stub"
    kind = "statistical"

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.fitted = True

    def score(self, windows: np.ndarray) -> np.ndarray:
        return np.full(len(windows), self.value, dtype="float64")

    def save(self, directory) -> None:  # pragma: no cover
        pass

    def load(self, directory) -> None:  # pragma: no cover
        pass

    @property
    def state_attrs(self) -> tuple:
        return ()


def synthetic_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {C.DATE_COLUMN: pd.date_range("2022-03-27 14:28:54", periods=n, freq="5s")}
    )
    for tag in C.TAGS:
        frame[tag] = rng.normal(100.0, 5.0, n)
    return frame


def build_agent(frame: pd.DataFrame, part_keys=("drum", "id_fan"), lookback: int = 8):
    scalers = fit_scalers(frame)
    ensembles = {}
    for key in part_keys:
        ensemble = PartEnsemble(
            part_key=key,
            lookback=lookback,
            tags=part_tags(key),
            detectors={},
            weights={"stub": 1.0},
            mode="single",
            best_key="stub",
            threshold=2.0,
        )
        ensemble.detectors["stub"] = StubDetector()
        ensemble.references["stub"] = (0.0, 1.0)
        ensembles[key] = ensemble
    return OverviewAgent(
        ensembles=ensembles,
        scalers=scalers,
        baselines=OverviewAgent._baselines(frame, scalers),
    )


@pytest.fixture
def client():
    frame = synthetic_frame()
    controller = ScenarioController(frame, mode="livestream", speed=1.0)
    agent = build_agent(frame)
    app = create_app(agent, controller)
    app.config["TESTING"] = True
    return app.test_client(), controller, agent


# --- pages ----------------------------------------------------------------


def test_boiler_page_renders_diagram(client):
    http, _controller, _agent = client
    response = http.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="boiler-svg"' in body
    for part_key in C.part_keys():
        assert f'data-part="{part_key}"' in body
    assert "Overview Agent" in body
    assert 'id="mode-select"' in body and 'id="scenario-select"' in body


def test_boiler_page_has_two_tabs(client):
    http, _controller, _agent = client
    body = http.get("/").get_data(as_text=True)
    assert 'data-tab="live"' in body and 'data-tab="history"' in body
    assert 'id="tab-live"' in body and 'id="tab-history"' in body


def test_history_tab_renders_one_panel_per_part(client):
    http, _controller, _agent = client
    body = http.get("/").get_data(as_text=True)
    # server-rendered score panels, so each part always gets its own axis
    assert body.count('class="history-panel" data-part=') == len(C.PARTS)
    for part_key in C.part_keys():
        assert f'class="history-panel" data-part="{part_key}"' in body
        assert f'<canvas class="panel-canvas" data-part="{part_key}"' in body
    assert 'id="plant-strip"' in body
    assert 'id="tag-panels"' in body
    assert 'id="focus-select"' in body


def test_inject_feedback_hooks_exist(client):
    """The Inject button must have somewhere to show progress and results."""
    http, _controller, _agent = client
    body = http.get("/").get_data(as_text=True)
    assert 'id="btn-start"' in body and 'class="spinner"' in body
    assert 'id="scenario-card"' in body
    assert 'id="scenario-name"' in body and 'id="scenario-meta"' in body
    assert 'id="scenario-truth"' in body and 'id="sev-fill"' in body
    assert 'id="btn-stop-2"' in body
    assert 'id="toast-host"' in body


def test_frontend_js_wires_inject_hold_and_panels(client):
    http, _controller, _agent = client
    js = http.get("/static/js/app.js").get_data(as_text=True)
    assert "/api/live" in js and "/api/situation" in js
    assert "hold: true" in js            # faults stay active until stopped
    assert "toast(" in js                # visible feedback
    assert "setBusy(" in js              # button busy state
    assert "drawPanel(" in js            # per-panel axes
    assert "switchTab(" in js


def test_scenario_dropdown_lists_demo_presets(client):
    http, _controller, _agent = client
    body = http.get("/").get_data(as_text=True)
    assert "demo_idf_degradation" in body
    assert "DEMO" in body


def test_part_page_renders_and_unknown_part_404(client):
    http, _controller, _agent = client
    ok = http.get("/part/id_fan")
    assert ok.status_code == 200
    body = ok.get_data(as_text=True)
    assert "Induced Draft Fan" in body
    assert "Monitored tags" in body
    assert http.get("/part/nonsense").status_code == 404


def test_static_assets_served(client):
    http, _controller, _agent = client
    css = http.get("/static/css/app.css")
    js = http.get("/static/js/app.js")
    assert css.status_code == 200 and b"--nominal" in css.data
    assert js.status_code == 200 and b"/api/live" in js.data


# --- live + situation payloads --------------------------------------------


def test_live_payload_structure(client):
    http, _controller, _agent = client
    payload = http.get("/api/live").get_json()
    assert set(payload) >= {
        "position",
        "timestamp",
        "mode",
        "speed",
        "scenario",
        "labels",
        "situation",
        "parts",
    }
    part = payload["parts"]["drum"]
    assert set(part) >= {
        "name", "level", "score", "threshold", "trend",
        "values", "deviations", "top_tags",
    }
    assert set(part["values"]) == set(part_tags("drum"))
    assert set(part["deviations"]) == set(part_tags("drum"))
    assert payload["situation"]["level"] in {"NOMINAL", "WATCH", "ALERT", "CRITICAL"}
    assert payload["situation"]["counts"]["NOMINAL"] >= 1


def test_situation_endpoint(client):
    http, _controller, _agent = client
    payload = http.get("/api/situation").get_json()
    assert "situation" in payload and "attention" in payload
    assert payload["situation"]["summary"]


def test_health_endpoint(client):
    http, _controller, _agent = client
    payload = http.get("/api/health").get_json()
    assert payload["status"] == "ok"
    assert set(payload["parts"]) == {"drum", "id_fan"}
    assert set(payload["bands"]["drum"]) == {"watch", "alert", "critical"}


def test_scenarios_endpoint_lists_all(client):
    http, _controller, _agent = client
    payload = http.get("/api/scenarios").get_json()
    assert len(payload["scenarios"]) >= 12
    assert "state" in payload
    assert payload["parts"]["drum"]["name"] == "Steam Drum & Pot"
    assert payload["interval_s"] == C.SAMPLE_INTERVAL_S


# --- control endpoints ----------------------------------------------------


def test_start_stop_scenario(client):
    http, controller, agent = client
    started = http.post("/api/scenarios/start", json={"scenario_key": "demo_water_wall_leak"})
    assert started.status_code == 200
    assert started.get_json()["state"]["scenario_key"] == "demo_water_wall_leak"
    assert controller.mode == "simulation"
    assert controller.active is not None

    stopped = http.post("/api/scenarios/stop")
    assert stopped.status_code == 200
    assert stopped.get_json()["state"]["scenario_key"] is None


def test_start_holds_fault_until_stopped(client):
    """The UI relies on hold=True so fast playback does not end the demo."""
    http, controller, _agent = client
    http.post("/api/scenarios/start", json={"scenario_key": "demo_idf_degradation"})
    duration = controller.active.duration_s
    assert duration == float("inf")
    # far beyond any preset duration the fault is still active
    assert controller.current_fault(100_000) is not None


def test_start_without_hold_uses_preset_duration(client):
    http, controller, _agent = client
    http.post(
        "/api/scenarios/start",
        json={"scenario_key": "demo_idf_degradation", "hold": False, "duration_s": 300},
    )
    assert controller.active.duration_s == 300.0
    assert controller.current_fault(controller.position + 100) is None


def test_unknown_scenario_rejected(client):
    http, _controller, _agent = client
    response = http.post("/api/scenarios/start", json={"scenario_key": "nope"})
    assert response.status_code == 400
    assert "unknown scenario" in response.get_json()["error"]


def test_mode_switch_and_validation(client):
    http, controller, _agent = client
    assert http.post("/api/mode", json={"mode": "simulation"}).status_code == 200
    assert controller.mode == "simulation"
    assert http.post("/api/mode", json={"mode": "livestream"}).status_code == 200
    assert controller.mode == "livestream"
    bad = http.post("/api/mode", json={"mode": "turbo"})
    assert bad.status_code == 400


def test_speed_endpoint_and_validation(client):
    http, controller, _agent = client
    assert http.post("/api/speed", json={"speed": 30}).status_code == 200
    assert controller.speed == 30.0
    assert http.post("/api/speed", json={"speed": 0}).status_code == 400
    assert http.post("/api/speed", json={"speed": "fast"}).status_code == 400


def test_reset_endpoint_rewinds_and_clears_state(client):
    http, controller, agent = client
    agent.states["drum"].level = 3
    response = http.post("/api/reset", json={"start_index": 10})
    assert response.status_code == 200
    assert controller.position == 10
    assert agent.states["drum"].level == 0


def test_live_payload_helper_advances_controller():
    frame = synthetic_frame()
    controller = ScenarioController(frame, mode="livestream", speed=10.0)
    agent = build_agent(frame)
    controller.tick(now=0.0)  # first call primes the clock
    payload = live_payload(agent, controller, advance=False)
    assert payload["position"] == controller.position
    assert scenario_state(controller)["mode"] == "livestream"


# --- JSON transport safety ------------------------------------------------


def _strict(text: str) -> dict:
    """Parse like a browser does: Infinity/NaN are *invalid* JSON."""

    def reject(constant: str):
        raise ValueError(f"invalid JSON constant on the wire: {constant}")

    return json.loads(text, parse_constant=reject)


def test_payloads_never_emit_infinity_or_nan(client):
    """Regression: a held fault has duration_s = inf, which Python prints as
    bare `Infinity` — invalid JSON that breaks the browser poller while
    json.loads (and Flask's test client) accept it silently.
    """
    http, controller, _agent = client
    http.post("/api/scenarios/start", json={"scenario_key": "demo_idf_degradation"})
    assert controller.active.duration_s == float("inf")  # held until stopped

    for path in ("/api/live", "/api/situation", "/api/scenarios", "/api/health"):
        raw = http.get(path).get_data(as_text=True)
        for bad in ("Infinity", "-Infinity", "NaN"):
            assert bad not in raw, f"{path} emitted {bad}"
        _strict(raw)  # raises if anything non-finite slipped through


def test_held_flag_is_reported_for_held_faults(client):
    http, _controller, _agent = client
    http.post("/api/scenarios/start", json={"scenario_key": "demo_water_wall_leak"})
    live = http.get("/api/live").get_json()
    assert live["scenario"]["held"] is True
    assert live["scenario"]["duration_s"] is None
    assert http.get("/api/scenarios").get_json()["state"]["held"] is True


def test_finite_duration_is_reported_when_not_held(client):
    http, _controller, _agent = client
    http.post(
        "/api/scenarios/start",
        json={"scenario_key": "demo_water_wall_leak", "hold": False, "duration_s": 600},
    )
    live = http.get("/api/live").get_json()
    assert live["scenario"]["held"] is False
    assert live["scenario"]["duration_s"] == 600
    for bad in ("Infinity", "NaN"):
        assert bad not in http.get("/api/live").get_data(as_text=True)


def test_json_safe_maps_non_finite_floats():
    from bhel_optimize.jsonio import json_safe

    payload = {"a": float("inf"), "b": [float("-inf"), float("nan"), 1.5], "c": {"d": 2}}
    safe = json_safe(payload)
    assert safe["a"] is None
    assert safe["b"][:2] == [None, None]
    assert safe["b"][2] == 1.5 and safe["c"]["d"] == 2
