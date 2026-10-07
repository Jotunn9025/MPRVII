"""The generated scenario documentation must stay in sync with the code."""

from __future__ import annotations

from bhel_optimize.simulation import faults as F
from bhel_optimize.simulation.report import markdown_report


def test_report_lists_every_fault_and_effect():
    report = markdown_report()
    for key, fault in F.FAULTS.items():
        assert f"`{key}`" in report
        for effect in fault.effects:
            assert f"`{effect.tag}`" in report


def test_report_lists_every_scenario():
    report = markdown_report()
    for key in F.SCENARIOS:
        assert f"`{key}`" in report


def test_report_marks_itself_generated_and_documents_gain_convention():
    report = markdown_report()
    assert "do not hand-edit" in report
    assert "multiples of each tag's IQR" in report


def test_committed_doc_matches_generated_report():
    """`.agents/scenarios.md` must equal the generated output."""
    from bhel_optimize import config as C

    path = C.PROJECT_ROOT / ".agents" / "scenarios.md"
    if not path.exists():
        return  # doc not present in this checkout
    assert path.read_text().strip() == markdown_report().strip(), (
        "regenerate with: uv run python -m bhel_optimize.simulation.report "
        "> .agents/scenarios.md"
    )
