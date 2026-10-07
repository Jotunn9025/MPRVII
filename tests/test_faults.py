"""Tests for the fault library: specs, propagation matrix, dynamics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bhel_optimize import config as C
from bhel_optimize.simulation import faults as F

VALID_KINDS = {"step", "ramp", "oscillate", "spike", "stuck"}


# --- spec integrity -------------------------------------------------------


def test_fault_keys_match_and_unique():
    assert F.FAULTS
    for key, spec in F.FAULTS.items():
        assert spec.key == key


def test_fault_parts_are_valid_and_effect_tags_exist():
    for spec in F.FAULTS.values():
        assert spec.part in C.PARTS, spec.key
        assert spec.effects, spec.key
        for effect in spec.effects:
            assert effect.tag in C.TAGS, (spec.key, effect.tag)
            assert effect.kind in VALID_KINDS


def test_fault_categories_are_known():
    assert {s.category for s in F.FAULTS.values()} <= {
        "creeping",
        "abrupt",
        "sensor",
        "global",
    }
    for category in ("creeping", "abrupt", "sensor", "global"):
        assert any(s.category == category for s in F.FAULTS.values()), category


def test_process_faults_touch_their_own_part():
    for spec in F.FAULTS.values():
        if spec.category in {"sensor", "global"}:
            continue
        assert any(e.tag in C.PARTS[spec.part].tags for e in spec.effects), spec.key


def test_propagation_matrix_is_multi_sensor_for_process_faults():
    """The roadmap requirement: faults must move correlated sensors too."""
    for spec in F.FAULTS.values():
        if spec.category == "sensor":
            assert len(spec.effects) == 1, "sensor faults are single-tag"
            continue
        if spec.category == "global":
            continue
        assert len(spec.effects) >= 3, spec.key
        parts = {C.part_of(e.tag) for e in spec.effects}
        assert len(parts) >= 2, f"{spec.key} only affects {parts}"


def test_sensor_faults_have_valid_target_tag():
    for key in ("sensor_outlier_spike", "sensor_stuck"):
        spec = F.FAULTS[key]
        assert spec.target_tag in C.TAGS
        assert spec.effects[0].tag == spec.target_tag
        assert spec.effects[0].kind in {"spike", "stuck"}


def test_target_tag_substitution():
    spec = F.FAULTS["sensor_outlier_spike"]
    effects = spec.effects_for("AIR_8301B.AV_0")
    assert effects[0].tag == "AIR_8301B.AV_0"
    assert effects[0].kind == "spike"
    with pytest.raises(ValueError):
        spec.effects_for("NOT_A_TAG")


def test_global_fault_targets_state_tag():
    spec = F.FAULTS["steam_temp_anomaly"]
    assert spec.part == "main_steam"
    assert any(e.tag == C.STATE_TAG for e in spec.effects)


def test_creeping_faults_use_ramp_dynamics():
    for key in (
        "economizer_fouling",
        "id_fan_degradation",
        "air_preheater_fouling",
        "bed_dp_drift",
    ):
        kinds = {e.kind for e in F.FAULTS[key].effects}
        assert "ramp" in kinds, key


def test_abrupt_faults_use_step_or_oscillate():
    for key in (
        "water_wall_tube_leak",
        "superheater_tube_leak",
        "id_fan_bearing_failure",
        "furnace_draft_disturbance",
    ):
        kinds = {e.kind for e in F.FAULTS[key].effects}
        assert kinds <= {"step"}, key
    assert {e.kind for e in F.FAULTS["attemperator_valve_stuck"].effects} <= {
        "stuck",
        "oscillate",
        "step",
    }


# --- dynamics -------------------------------------------------------------


def test_response_zero_before_dead_time():
    assert F.response("step", 0.0, dead_time_s=10.0) == 0.0
    assert F.response("step", 9.9, dead_time_s=10.0) == 0.0


def test_step_response_monotonic_and_bounded():
    values = [F.response("step", t, tau_s=30.0) for t in range(0, 300, 10)]
    assert values[0] == 0.0
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert max(values) <= 1.0
    assert max(values) > 0.99


def test_ramp_response_linear_then_saturates():
    assert F.response("ramp", 100.0, ramp_s=400.0) == pytest.approx(0.25)
    assert F.response("ramp", 400.0, ramp_s=400.0) == 1.0
    assert F.response("ramp", 9999.0, ramp_s=400.0) == 1.0


def test_oscillate_response_signed():
    period = 100.0
    assert F.response("oscillate", 25.0, period_s=period) == pytest.approx(1.0)
    assert F.response("oscillate", 75.0, period_s=period) == pytest.approx(-1.0)
    assert F.response("oscillate", 100.0, period_s=period) == pytest.approx(0.0, abs=1e-9)


def test_spike_response_only_inside_window():
    assert F.response("spike", 1.0, width_s=10.0) == 1.0
    assert F.response("spike", 10.0, width_s=10.0) == 1.0
    assert F.response("spike", 11.0, width_s=10.0) == 0.0


def test_stuck_response_is_full_hold():
    assert F.response("stuck", 5.0) == 1.0


def test_unknown_response_kind_raises():
    with pytest.raises(ValueError):
        F.response("magic", 1.0)


# --- scales, scenarios, docs ---------------------------------------------


def test_tag_scales_are_positive_and_correct():
    frame = pd.DataFrame({tag: [0.0, 1.0, 2.0, 3.0, 4.0] for tag in C.TAGS})
    scales = F.tag_scales(frame)
    assert set(scales) == set(C.TAGS)
    assert all(v > 0 for v in scales.values())
    assert scales[C.TAGS[0]] == pytest.approx(2.0)  # IQR of 0..4 is 2


def test_scenarios_reference_real_faults_and_are_unique():
    assert len(F.SCENARIOS) == len(set(F.SCENARIOS))
    for key, scenario in F.SCENARIOS.items():
        assert scenario.key == key
        assert scenario.fault_key in F.FAULTS
        assert scenario.duration_s > 0


def test_demo_presets_exist():
    demos = [s for s in F.SCENARIOS.values() if s.demo]
    assert len(demos) == 4
    assert sum(1 for s in demos if s.spec().category == "creeping") == 1
    assert sum(1 for s in demos if s.spec().category == "abrupt") == 1


def test_list_scenarios_shape():
    listed = F.list_scenarios()
    assert len(listed) == len(F.SCENARIOS)
    assert listed[0]["demo"] is True
    for entry in listed:
        assert set(entry) >= {
            "key",
            "name",
            "fault_key",
            "category",
            "part",
            "duration_s",
            "description",
        }


def test_describe_faults_documents_propagation():
    described = {d["key"]: d for d in F.describe_faults()}
    assert set(described) == set(F.FAULTS)
    for entry in described.values():
        assert entry["part_name"] == C.PARTS[entry["part"]].name
        assert entry["effects"]
        for effect in entry["effects"]:
            assert effect["tag_description"] == C.TAG_INFO[effect["tag"]]
