"""Tests for the Overview Agent: levels, trends, patterns, payload, API."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bhel_optimize import config as C
from bhel_optimize.data import fit_scalers
from bhel_optimize.detection.ensembles import PartEnsemble
from bhel_optimize.detection.windows import part_tags
from bhel_optimize.overview.agent import OverviewAgent
from bhel_optimize.overview.api import create_overview_app, situation_payload
from bhel_optimize.overview.situation import (
    ALERT,
    CRITICAL,
    LEVEL_NAMES,
    NOMINAL,
    WATCH,
    PartState,
    apply_persistence,
    criticality_adjusted,
    criticality_escalated,
    level_for_score,
    match_fault_patterns,
    plant_level,
    step_level,
    top_deviating_tags,
    trend_for_history,
)


# --------------------------------------------------------------------------
# level model
# --------------------------------------------------------------------------


def test_level_for_score_bands():
    threshold = 2.0
    assert level_for_score(0.0, threshold) == NOMINAL
    assert level_for_score(0.5 * threshold, threshold) == NOMINAL
    assert level_for_score(0.6 * threshold, threshold) == WATCH
    assert level_for_score(threshold, threshold) == ALERT
    assert level_for_score(3.0 * threshold, threshold) == CRITICAL


def test_level_for_score_handles_bad_thresholds_and_nan():
    assert level_for_score(float("nan"), 2.0) == NOMINAL
    assert level_for_score(10.0, 0.0) == NOMINAL
    assert level_for_score(10.0, float("nan")) == NOMINAL
    assert level_for_score(1.0, 0.0) == NOMINAL


def test_persistence_downgrades_fresh_alerts():
    state = PartState(part_key="drum")
    assert apply_persistence(ALERT, state) == WATCH  # nothing persisted yet
    for _ in range(3):
        state.above_flags.append(True)
    assert apply_persistence(ALERT, state) == ALERT
    assert apply_persistence(CRITICAL, state) == CRITICAL
    assert apply_persistence(NOMINAL, state) == NOMINAL


def test_step_level_is_rate_limited():
    assert step_level(NOMINAL, CRITICAL) == WATCH
    assert step_level(WATCH, CRITICAL) == ALERT
    assert step_level(ALERT, CRITICAL) == CRITICAL
    assert step_level(CRITICAL, NOMINAL) == ALERT


def test_trend_detection():
    threshold = 2.0
    rising = list(np.linspace(0.0, 2.0, 12))
    falling = list(np.linspace(2.0, 0.0, 12))
    steady = [1.0] * 12
    assert trend_for_history(rising, threshold) == "rising"
    assert trend_for_history(falling, threshold) == "falling"
    assert trend_for_history(steady, threshold) == "steady"
    assert trend_for_history([1.0], threshold) == "steady"


def test_criticality_escalation_and_plant_level():
    assert criticality_adjusted(WATCH, 5, held_s=600) == ALERT
    assert criticality_adjusted(WATCH, 2, held_s=600) == WATCH
    assert criticality_adjusted(WATCH, 5, held_s=10) == WATCH  # not sustained yet
    assert criticality_adjusted(ALERT, 5, held_s=600) == ALERT
    # plant level is the worst part level: an ALERT must mean a part really
    # crossed its threshold, so normal noise on a critical part cannot lift it
    assert plant_level({"drum": WATCH, "return_air": NOMINAL}) == WATCH
    assert plant_level({"return_air": WATCH, "economizer": NOMINAL}) == WATCH
    assert plant_level({"id_fan": CRITICAL}) == CRITICAL
    assert plant_level({}) == NOMINAL
    # ...criticality instead surfaces as an emphasis flag for the summary
    assert criticality_escalated({"drum": WATCH}, {"drum": 600}) == ["drum"]
    assert criticality_escalated({"drum": WATCH}, {"drum": 10}) == []


def test_top_deviating_tags_ranks_by_magnitude():
    deviations = {"a": 0.2, "b": -3.0, "c": 1.1, "d": float("nan")}
    assert [tag for tag, _ in top_deviating_tags(deviations, limit=2)] == ["b", "c"]


def test_level_names_are_exposed_in_order():
    assert LEVEL_NAMES == ("NOMINAL", "WATCH", "ALERT", "CRITICAL")


# --------------------------------------------------------------------------
# pattern matcher
# --------------------------------------------------------------------------


def test_pattern_matcher_finds_idf_degradation():
    deviations = {
        "YFJ3_ZD1.AV_0": 0.9,
        "YFJ3_ZD2.AV_0": 0.8,
        "YFJ3_AI.AV_0": 0.6,
    }
    matches = match_fault_patterns("id_fan", deviations)
    assert matches, "expected at least one pattern match"
    assert matches[0]["fault_key"] == "id_fan_degradation"
    assert matches[0]["confidence"] > 0.6


def test_pattern_matcher_prefers_abrupt_fault_for_large_deviations():
    """A spike-sized deviation fits the abrupt bearing-failure signature."""
    deviations = {
        "YFJ3_ZD1.AV_0": 2.0,
        "YFJ3_ZD2.AV_0": 1.6,
        "YFJ3_AI.AV_0": 1.0,
    }
    matches = match_fault_patterns("id_fan", deviations)
    assert matches and matches[0]["fault_key"] == "id_fan_bearing_failure"


def test_pattern_matcher_prefers_creep_for_gradual_deviations():
    """Same tags, gradual magnitudes -> the creeping degradation signature."""
    deviations = {
        "YFJ3_ZD1.AV_0": 0.9,
        "YFJ3_ZD2.AV_0": 0.8,
        "YFJ3_AI.AV_0": 0.6,
    }
    matches = match_fault_patterns("id_fan", deviations)
    assert matches and matches[0]["fault_key"] == "id_fan_degradation"
    assert all(m["fault_key"] != "id_fan_bearing_failure" for m in matches)


def test_pattern_matcher_ignores_faults_centred_elsewhere():
    """A single coupled tag must not make another part's fault look local."""
    from bhel_optimize.overview.situation import PATTERN_MIN_CONFIDENCE

    matches = match_fault_patterns("id_fan", {"YFJ3_AI.AV_0": 20.0})
    assert all(m["fault_key"] != "economizer_fouling" for m in matches)
    assert all(m["confidence"] >= PATTERN_MIN_CONFIDENCE for m in matches)


def test_pattern_matcher_respects_direction():
    deviations = {"YFJ3_ZD1.AV_0": -0.9, "YFJ3_ZD2.AV_0": -0.8}
    matches = match_fault_patterns("id_fan", deviations)
    assert all(m["fault_key"] != "id_fan_degradation" for m in matches)


def test_pattern_matcher_ignores_small_deviations():
    assert match_fault_patterns("id_fan", {"YFJ3_ZD1.AV_0": 0.05}) == []


# --------------------------------------------------------------------------
# agent (with stub detectors -> deterministic scores)
# --------------------------------------------------------------------------


class StubDetector:
    """Detector stub returning a scripted score per window."""

    key = "stub"
    name = "Stub"
    kind = "statistical"

    def __init__(self) -> None:
        self.value = 0.0
        self.fitted = True

    def score(self, windows: np.ndarray) -> np.ndarray:
        return np.full(len(windows), self.value, dtype="float64")

    def save(self, directory) -> None:  # pragma: no cover - interface parity
        pass

    def load(self, directory) -> None:  # pragma: no cover - interface parity
        pass


def _synthetic_frame(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({C.DATE_COLUMN: pd.date_range("2022-01-01", periods=n, freq="5s")})
    for tag in C.TAGS:
        frame[tag] = rng.normal(100.0, 5.0, n)
    return frame


def _agent(part_keys: tuple[str, ...] = ("drum", "id_fan"), lookback: int = 5, threshold: float = 2.0):
    frame = _synthetic_frame()
    scalers = fit_scalers(frame)
    ensembles = {}
    stubs = {}
    for key in part_keys:
        ensemble = PartEnsemble(
            part_key=key,
            lookback=lookback,
            tags=part_tags(key),
            detectors={},
            references={},
            weights={"stub": 1.0},
            mode="single",
            best_key="stub",
            threshold=threshold,
        )
        stub = StubDetector()
        ensemble.detectors["stub"] = stub
        ensemble.references["stub"] = (0.0, 1.0)
        ensembles[key] = ensemble
        stubs[key] = stub
    agent = OverviewAgent(
        ensembles=ensembles,
        scalers=scalers,
        baselines=OverviewAgent._baselines(frame, scalers),
    )
    return agent, stubs, frame


def test_agent_warmup_reports_nominal_without_enough_history():
    agent, _stubs, frame = _agent(lookback=20)
    overview = agent.update(frame, 0)
    assert overview["parts"]["drum"]["level"] == "NOMINAL"
    assert overview["parts"]["drum"]["score"] is None
    assert overview["situation"]["level"] == "NOMINAL"


def test_agent_overview_shape_and_nominal_state():
    agent, _stubs, frame = _agent()
    overview = agent.update(frame, 20)
    assert overview["situation"]["level"] == "NOMINAL"
    assert set(overview["parts"]) == {"drum", "id_fan"}
    part = overview["parts"]["drum"]
    assert set(part) >= {
        "level",
        "score",
        "threshold",
        "trend",
        "top_tags",
        "severity_iqr",
        "pattern",
        "explanation",
    }
    assert "nominal" in part["explanation"].lower()
    assert json.loads(agent.to_json(overview))["situation"]["level"] == "NOMINAL"


def test_agent_escalates_with_sustained_high_scores():
    agent, stubs, frame = _agent()
    stubs["drum"].value = 3.0  # above threshold
    levels = []
    for index in range(15, 15 + 12):
        overview = agent.update(frame, index)
        levels.append(overview["parts"]["drum"]["level"])
    assert levels[-1] in {"ALERT", "CRITICAL"}
    assert agent.states["drum"].history
    part = agent.update(frame, 40)["parts"]["drum"]
    assert part["duration_s"] > 0
    assert part["severity_iqr"] >= 0
    assert "review" in part["explanation"].lower()


def test_agent_critical_needs_extreme_score():
    agent, stubs, frame = _agent()
    stubs["drum"].value = 5.0  # >= 2x threshold
    for index in range(10, 30):
        overview = agent.update(frame, index)
    assert overview["parts"]["drum"]["level"] == "CRITICAL"
    assert overview["situation"]["level"] == "CRITICAL"
    assert "critical" in overview["situation"]["summary"].lower()


def test_agent_criticality_is_reported_without_inflating_level():
    agent, stubs, frame = _agent(part_keys=("drum",))
    stubs["drum"].value = 2.5  # above threshold -> ALERT, sustained
    for index in range(10, 40):
        overview = agent.update(frame, index)
    assert overview["parts"]["drum"]["level"] in {"ALERT", "CRITICAL"}
    assert overview["situation"]["level"] == overview["parts"]["drum"]["level"]
    assert overview["situation"]["counts"][overview["situation"]["level"]] >= 1


def test_agent_decays_after_fault_clears():
    agent, stubs, frame = _agent()
    stubs["id_fan"].value = 4.0
    for index in range(10, 25):
        agent.update(frame, index)
    assert agent.states["id_fan"].level >= ALERT
    stubs["id_fan"].value = 0.0
    for index in range(25, 45):
        overview = agent.update(frame, index)
    assert overview["parts"]["id_fan"]["level"] == "NOMINAL"


def test_agent_reset_clears_state():
    agent, stubs, frame = _agent()
    stubs["drum"].value = 4.0
    for index in range(10, 20):
        agent.update(frame, index)
    assert agent.states["drum"].level > NOMINAL
    agent.reset()
    assert agent.states["drum"].level == NOMINAL
    assert not agent.states["drum"].history


def test_agent_reports_probable_pattern_and_top_tags():
    agent, stubs, frame = _agent(part_keys=("id_fan",))
    # force a large positive deviation on the IDF tags the matcher expects
    frame.loc[30, ["YFJ3_ZD1.AV_0", "YFJ3_ZD2.AV_0", "YFJ3_AI.AV_0"]] = [200.0, 200.0, 200.0]
    stubs["id_fan"].value = 4.0
    overview = agent.update(frame, 30)
    part = overview["parts"]["id_fan"]
    assert part["top_tags"][0]["direction"] == "above"
    assert part["pattern"] is not None
    # the matcher distinguishes creep from abrupt faults by magnitude, so both
    # IDF patterns are acceptable explanations for an artificial step change
    assert part["pattern"]["fault_key"] in {
        "id_fan_degradation",
        "id_fan_bearing_failure",
    }
    assert "likely" in part["explanation"].lower()


def test_agent_attention_list_is_ordered_by_severity():
    agent, stubs, frame = _agent(part_keys=("drum", "id_fan", "economizer"))
    stubs["id_fan"].value = 5.0
    stubs["drum"].value = 1.4
    for index in range(10, 25):
        overview = agent.update(frame, index)
    names = [item["key"] for item in overview["attention"]]
    assert names[0] == "id_fan"
    assert "drum" in names


def test_agent_save_baselines(tmp_path):
    agent, _stubs, _frame = _agent()
    path = agent.save(tmp_path / "overview.json")
    payload = json.loads(path.read_text())
    assert set(payload["baselines"]) == set(C.TAGS)
    assert payload["parts"] == ["drum", "id_fan"]


# --------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------


class StubController:
    """Minimal stand-in for ScenarioController for API tests."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.position = 20
        self.mode = "simulation"
        self.speed = 1.0

    def tick(self, now: float | None = None) -> int:  # pragma: no cover - not advanced
        return 0

    def snapshot(self, index: int | None = None) -> dict:
        return {
            "mode": self.mode,
            "position": self.position,
            "total_samples": len(self.frame),
            "speed": self.speed,
            "at_end": False,
            "scenario": {"scenario_key": "demo_x", "name": "Demo", "part": "drum"},
            "labels": {"fault_key": None, "fault_part": None, "fault_active": False},
        }


def test_situation_endpoint_returns_json():
    agent, _stubs, frame = _agent()
    app = create_overview_app(agent, StubController(frame))
    client = app.test_client()

    response = client.get("/api/situation")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["situation"]["level"] in LEVEL_NAMES
    assert "parts" in payload and "drum" in payload["parts"]
    assert payload["mode"] == "simulation"
    assert payload["scenario"]["scenario_key"] == "demo_x"

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"
    assert set(health.get_json()["parts"]) == {"drum", "id_fan"}


def test_situation_payload_helper_without_advancing():
    agent, _stubs, frame = _agent()
    controller = StubController(frame)
    payload = situation_payload(agent, controller)
    assert payload["position"] == controller.position
    assert payload["parts"]["drum"]["level"] == "NOMINAL"


# --------------------------------------------------------------------------
# integration with real artifacts (skipped when not trained)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_agent_end_to_end_with_trained_artifacts():
    from bhel_optimize.detection.ensembles import ensemble_dir
    from bhel_optimize.simulation import ScenarioController

    if not (ensemble_dir("id_fan") / "ensemble.json").exists():
        pytest.skip("trained ensembles not present — run the detection trainer")

    agent = OverviewAgent.from_artifacts()
    controller = ScenarioController(mode="simulation", start_index=0, speed=1.0)
    controller.start_scenario("demo_idf_degradation")
    overview = None
    for index in range(120, 900, 15):  # 75 minutes of simulated time
        overview = agent.update(controller.frame, index)
    assert overview is not None
    id_fan = overview["parts"]["id_fan"]
    assert id_fan["level_index"] >= WATCH
    assert overview["situation"]["level_index"] >= WATCH
    assert id_fan["score"] > 0


# --------------------------------------------------------------------------
# Groq narration (no network: the HTTP call is stubbed)
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str, status: int = 200) -> None:
        self._text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._text}}]}


def _narrator(**kwargs):
    from bhel_optimize.overview.llm import GroqNarrator

    return GroqNarrator(api_key="test-key", min_interval_s=0.0, **kwargs)


def test_narrator_disabled_without_key(monkeypatch):
    from bhel_optimize.overview.llm import GroqNarrator

    narrator = GroqNarrator(api_key=None)
    assert narrator.available is False
    assert narrator.summarise({"plant_level": "NOMINAL"}) == (None, "template")


def test_narrator_returns_model_text(monkeypatch):
    import requests

    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        return _FakeResponse("Boiler steady; drum drifting slightly high.")

    monkeypatch.setattr(requests, "post", fake_post)
    narrator = _narrator()
    text, source = narrator.summarise({"plant_level": "WATCH", "parts": []})
    assert source == "groq"
    assert text.startswith("Boiler steady")
    assert narrator.calls == 1
    # the request carries the key, the model and a bounded output budget
    body = calls[0]["json"]
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert body["model"] and body["max_tokens"] <= 200
    assert "two short sentences" in body["messages"][0]["content"].lower()


def test_narrator_caches_identical_situations(monkeypatch):
    import requests

    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        return _FakeResponse("Same situation.")

    monkeypatch.setattr(requests, "post", fake_post)
    narrator = _narrator()
    facts = {"plant_level": "NOMINAL", "parts": []}
    assert narrator.summarise(facts)[1] == "groq"
    assert narrator.summarise(facts)[1] == "groq"
    assert len(calls) == 1, "identical facts must not hit the API twice"


def test_narrator_falls_back_on_error(monkeypatch):
    import requests

    def boom(url, **kwargs):
        raise TimeoutError("groq timeout")

    monkeypatch.setattr(requests, "post", boom)
    narrator = _narrator()
    text, source = narrator.summarise({"plant_level": "ALERT"})
    assert (text, source) == (None, "template")
    assert "TimeoutError" in (narrator.last_error or "")


def test_narrator_falls_back_on_empty_completion(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda url, **kw: _FakeResponse("   "))
    narrator = _narrator()
    assert narrator.summarise({"plant_level": "WATCH"}) == (None, "template")


def test_agent_uses_narrator_when_available():
    agent, _stubs, frame = _agent(part_keys=("drum",))
    agent.narrator = _narrator()
    agent.narrator._request = lambda facts: "Drum is steady."  # type: ignore[assignment]
    overview = agent.update(frame, 20)
    assert overview["situation"]["summary"] == "Drum is steady."
    assert overview["situation"]["narrative_source"] == "groq"
    assert overview["situation"]["facts"]["plant_level"] == "NOMINAL"


def test_agent_reports_simulated_fault_in_facts():
    agent, _stubs, frame = _agent(part_keys=("drum",))
    context = {"fault_part": "drum", "fault_key": "water_wall_tube_leak", "fault_active": True,
               "tailed_observed": False}
    overview = agent.update(frame, 20, context=context)
    facts = overview["situation"]["facts"]
    assert facts["simulated_fault"]["part"] == "Steam Drum & Pot"
    assert facts["simulated_fault"]["active"] is True
