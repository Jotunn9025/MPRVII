"""Simulation layer: playback modes, scenarios and fault injection."""

from bhel_optimize.simulation.controller import (
    MODES,
    ActiveFault,
    ScenarioController,
    parts_metadata,
    run_series,
)
from bhel_optimize.simulation.faults import (
    FAULTS,
    SCENARIOS,
    FaultEffect,
    FaultSpec,
    Scenario,
    default_tag_scales,
    describe_faults,
    list_scenarios,
    response,
    tag_scales,
)

__all__ = [
    "MODES",
    "ActiveFault",
    "FAULTS",
    "SCENARIOS",
    "FaultEffect",
    "FaultSpec",
    "Scenario",
    "ScenarioController",
    "default_tag_scales",
    "describe_faults",
    "list_scenarios",
    "parts_metadata",
    "response",
    "run_series",
    "tag_scales",
]
