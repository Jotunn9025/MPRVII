"""Tests for the scenario controller: modes, playback, injection, labels."""

from __future__ import annotations

import pytest

from bhel_optimize import config as C
from bhel_optimize.simulation import ScenarioController, faults as F, parts_metadata, run_series
from conftest import requires_data


@pytest.fixture
def controller(boiler_frame):
    """Fresh controller per test (no cross-test state leakage)."""
    return ScenarioController(boiler_frame, speed=1.0, start_index=0)


# --- construction / validation -------------------------------------------


def test_rejects_bad_mode_and_speed():
    with pytest.raises(ValueError):
        ScenarioController(mode="nope")
    with pytest.raises(ValueError):
        ScenarioController(speed=0)


def test_rejects_unknown_scenario_and_fault(controller):
    with pytest.raises(KeyError):
        controller.start_scenario("does-not-exist")
    with pytest.raises(KeyError):
        controller.start_fault("does-not-exist")


# --- livestream mode ------------------------------------------------------


@requires_data
def test_livestream_has_no_injection(controller):
    controller.reset()
    controller.set_mode("livestream")
    assert controller.current_fault() is None
    snapshot = controller.snapshot()
    assert snapshot["mode"] == "livestream"
    assert snapshot["deltas"] == {}
    assert snapshot["values"] == snapshot["baseline_values"]


@requires_data
def test_livestream_labels_track_dataset_rule(controller):
    controller.reset()
    low, high = C.NORMAL_STEAM_TEMP_C
    for index in (0, 10, 100):
        labels = controller.labels_at(index)
        steam = controller.frame.iloc[index][C.STATE_TAG]
        assert labels["tailed_real"] is bool(steam < low or steam > high)
        assert labels["fault_key"] is None
        assert labels["fault_active"] is False


@requires_data
def test_simulation_mode_without_fault_still_clean(controller):
    controller.reset()
    controller.set_mode("simulation")
    assert controller.snapshot()["deltas"] == {}
    assert controller.severity_at(0) == 0.0


# --- playback -------------------------------------------------------------


@requires_data
def test_advance_is_deterministic_and_clamped(controller):
    controller.reset(0)
    assert controller.advance(0) == 0
    assert controller.advance(5) == 5
    assert controller.advance(10) == 15
    controller.position = len(controller) - 3
    assert controller.advance(100) == len(controller) - 1
    assert controller.at_end


@requires_data
def test_tick_uses_wall_clock(controller):
    controller.reset()
    controller.set_speed(10.0)  # samples per second
    assert controller.tick(now=1000.0) == 0  # first call only sets the clock
    assert controller.tick(now=1000.5) == 5
    assert controller.tick(now=1000.5) == 0


@requires_data
def test_loop_wraps_at_end():
    controller = ScenarioController(speed=1.0, loop=True, start_index=0)
    controller.position = len(controller) - 2
    controller._last_tick = 100.0
    controller.tick(now=100.0 + 10)
    assert controller.position < len(controller) - 2


@requires_data
def test_reset_clears_fault(controller):
    controller.start_fault("id_fan_bearing_failure")
    assert controller.current_fault() is not None
    controller.reset()
    assert controller.current_fault() is None


# --- injection behaviour --------------------------------------------------


@requires_data
def test_start_scenario_switches_to_simulation_and_affects_primary_tag(controller):
    controller.reset(0)
    controller.start_scenario("demo_water_wall_leak")
    assert controller.mode == "simulation"
    assert controller.snapshot()["deltas"] == {}  # dead time not elapsed yet
    index = 60  # t = 300 s, inside the 600 s fault window
    deltas = controller.effects_at(index)
    assert C.STATE_TAG in deltas
    assert deltas[C.STATE_TAG] < 0, "steam temperature must fall on a tube leak"
    assert all(deltas[t] > 0 for t in C.PARTS["furnace"].tags if t.startswith("PT_"))


@requires_data
def test_injected_fault_propagates_across_parts(controller):
    """Roadmap requirement: correlated sensors move, not just one tag."""
    controller.reset(0)
    controller.start_scenario("demo_idf_degradation")
    index = 500  # t = 2500 s into a creeping fault
    deltas = controller.effects_at(index)
    parts = {C.part_of(tag) for tag in deltas}
    assert {"id_fan", "furnace", "economizer"} <= parts
    assert deltas["YFJ3_ZD1.AV_0"] > 0
    assert deltas["YFJ3_AI.AV_0"] > 0


@requires_data
def test_creeping_fault_grows_monotonically(controller):
    controller.reset(0)
    controller.start_scenario("demo_idf_degradation")  # 2700 s = 540 samples
    severity = [controller.severity_at(i) for i in (50, 200, 400, 520)]
    assert all(b >= a for a, b in zip(severity, severity[1:]))
    assert severity[-1] > severity[0]


@requires_data
def test_fault_expires_after_duration(controller):
    controller.reset(0)
    controller.start_scenario("demo_water_wall_leak", duration_s=100.0)
    inside = 30  # t = 150 s > 100 s duration
    assert controller.current_fault(inside) is None
    assert controller.effects_at(inside) == {}
    assert controller.snapshot(inside)["scenario"] is None
    assert controller.current_fault(10) is not None


@requires_data
def test_labels_mark_active_fault_and_part(controller):
    controller.reset(0)
    controller.start_scenario("demo_water_wall_leak")
    late = controller.labels_at(60)  # t = 300 s (within the 600 s window)
    assert late["fault_key"] == "water_wall_tube_leak"
    assert late["fault_part"] == "furnace"
    assert late["fault_category"] == "abrupt"
    assert late["fault_active"] is True
    assert late["fault_severity"] > 0.5
    early = controller.labels_at(0)
    assert early["fault_active"] is False


@requires_data
def test_stop_scenario_clears_injection(controller):
    controller.reset(0)
    controller.start_scenario("demo_water_wall_leak")
    assert controller.effects_at(60)
    controller.stop_scenario()
    assert controller.effects_at(60) == {}
    assert controller.mode == "simulation"


# --- sensor-specific faults ----------------------------------------------


@requires_data
def test_sensor_spike_affects_only_target_within_window(controller):
    controller.reset(0)
    controller.start_fault("sensor_outlier_spike", target_tag="AIR_8301B.AV_0")
    during = controller.effects_at(2)  # t = 10 s, inside the 10 s spike window
    assert set(during) == {"AIR_8301B.AV_0"}
    assert during["AIR_8301B.AV_0"] > 0
    after = controller.effects_at(20)  # t = 100 s, window over
    assert after == {}


@requires_data
def test_sensor_stuck_freezes_value_at_onset(controller):
    controller.reset(0)
    onset_value = float(controller.frame.iloc[0]["PT_8313A.AV_0"])
    controller.start_fault("sensor_stuck", target_tag="PT_8313A.AV_0")
    for index in (5, 50, 100):  # inside the 600 s duration
        observed = controller.values_at(index)["PT_8313A.AV_0"]
        assert observed == pytest.approx(onset_value)
        baseline = float(controller.frame.iloc[index]["PT_8313A.AV_0"])
        assert observed != baseline


@requires_data
def test_oscillating_fault_changes_sign(controller):
    controller.reset(0)
    controller.start_scenario("attemperator_valve_stuck")
    deltas = [controller.effects_at(i).get(C.STATE_TAG, 0.0) for i in range(20, 120, 6)]
    assert max(deltas) > 0
    assert min(deltas) < 0


@requires_data
def test_global_fault_drives_steam_temp_out_of_band(controller):
    controller.reset(0)
    assert controller.labels_at(0)["tailed_real"] is False
    controller.start_scenario("demo_tailed_steam_temp")  # 900 s = 180 samples
    labels = controller.labels_at(100)  # t = 500 s
    steam = controller.values_at(100)[C.STATE_TAG]
    low, high = C.NORMAL_STEAM_TEMP_C
    assert steam < low or steam > high
    assert labels["tailed_observed"] is True
    assert labels["tailed_real"] is False
    assert labels["fault_category"] == "global"


# --- payloads -------------------------------------------------------------


@requires_data
def test_snapshot_payload_shape(controller):
    controller.reset(0)
    controller.start_scenario("demo_idf_degradation")
    snapshot = controller.snapshot(100)
    assert set(snapshot) >= {
        "mode",
        "position",
        "total_samples",
        "speed",
        "timestamp",
        "values",
        "baseline_values",
        "deltas",
        "affected_tags",
        "labels",
        "scenario",
    }
    assert set(snapshot["values"]) == set(C.TAGS)
    assert snapshot["total_samples"] == len(controller)
    assert snapshot["scenario"]["part_name"] == "Induced Draft Fan"
    assert snapshot["affected_tags"] == sorted(snapshot["deltas"])


@requires_data
def test_status_payload(controller):
    controller.reset(0)
    controller.start_scenario("demo_idf_degradation")
    status = controller.status()
    assert status["mode"] == "simulation"
    assert status["scenario_key"] == "demo_idf_degradation"
    assert status["fault_part"] == "id_fan"
    assert status["elapsed_s"] == 0.0


def test_parts_metadata_shape():
    parts = parts_metadata()
    assert len(parts) == 10
    for key, part in parts.items():
        assert part["key"] == key
        assert part["name"] == C.PARTS[key].name
        assert part["criticality"] == C.PARTS[key].criticality
        assert part["tags"] == list(C.PARTS[key].tags)


# --- offline labelled series ---------------------------------------------


@requires_data
def test_run_series_is_deterministic_and_labelled():
    first = run_series("demo_water_wall_leak", samples=120)
    second = run_series("demo_water_wall_leak", samples=120)
    assert first.shape == second.shape
    assert first.equals(second)
    assert int(first["fault_active"].sum()) > 0
    assert set(first["fault_part"].dropna().unique()) == {"furnace"}
    assert set(C.TAGS) <= set(first.columns)
    assert not first[list(C.TAGS)].isna().to_numpy().any()


@requires_data
def test_run_series_respects_sample_count_and_start():
    frame = run_series("demo_sensor_spike", samples=25, start_index=1000)
    assert len(frame) == 25
    assert frame["fault_key"].iloc[0] == "sensor_outlier_spike"


@requires_data
def test_run_series_global_fault_produces_tailed_samples():
    frame = run_series("demo_tailed_steam_temp", samples=400)
    assert frame["tailed_observed"].any()
