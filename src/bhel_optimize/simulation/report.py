"""Render the fault library / propagation matrix as Markdown.

Usage::

    uv run python -m bhel_optimize.simulation.report > .agents/scenarios.md

Keeping the documentation generated means it cannot drift from the fault
definitions (a test asserts every fault appears in the output).
"""

from __future__ import annotations

from bhel_optimize import config as C
from bhel_optimize.simulation import faults as F


def markdown_report() -> str:
    """Full Markdown documentation of scenarios, faults and coupling."""
    lines: list[str] = []
    lines.append("# Onboarding Knowledge — Scenario & Fault Library")
    lines.append("")
    lines.append(
        "> **Generated** by `uv run python -m bhel_optimize.simulation.report` —"
        " do not hand-edit; change `simulation/faults.py` instead."
    )
    lines.append("")
    lines.append(
        "Gains are **multiples of each tag's IQR** measured from the real data"
        " (engineering units are not published), so `0.9` means"
        " \"0.9 x typical spread of that tag\"."
    )
    lines.append("")

    lines.append("## Demo scenarios (UI dropdown)")
    lines.append("")
    lines.append("| Key | Name | Category | Part | Duration | Description |")
    lines.append("|---|---|---|---|---|---|")
    for entry in F.list_scenarios():
        if not entry["demo"]:
            continue
        lines.append(
            f"| `{entry['key']}` | {entry['name']} | {entry['category']} |"
            f" {C.PARTS[entry['part']].name} | {entry['duration_s']:.0f} s |"
            f" {entry['description']} |"
        )
    lines.append("")

    lines.append("## Fault library and propagation matrix")
    lines.append("")
    for fault in F.FAULTS.values():
        lines.append(f"### `{fault.key}` — {fault.name}")
        lines.append("")
        lines.append(f"- **Category:** {fault.category}")
        lines.append(
            f"- **Primary part:** {C.PARTS[fault.part].name}"
            f" (criticality {C.PARTS[fault.part].criticality})"
        )
        lines.append(f"- **Default duration:** {fault.default_duration_s:.0f} s")
        lines.append(f"- **Description:** {fault.description}")
        if fault.target_tag:
            lines.append(
                f"- **Default target tag:** `{fault.target_tag}`"
                " (overridable per run)"
            )
        lines.append(f"- **Physics:** {fault.physics}")
        lines.append("")
        lines.append("| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |")
        lines.append("|---|---|---|---|---|---|---|")
        for effect in fault.effects:
            note = effect.note or ("primary" if effect.gain else "")
            lines.append(
                f"| `{effect.tag}` | {C.TAG_INFO[effect.tag]} |"
                f" {effect.gain:+.2f} | {effect.kind} | {effect.dead_time_s:.0f} s |"
                f" {effect.tau_s:.0f} s | {note} |"
            )
        lines.append("")

    lines.append("## Scenario presets (all)")
    lines.append("")
    lines.append("| Key | Name | Category | Duration | Demo |")
    lines.append("|---|---|---|---|---|")
    for entry in F.list_scenarios():
        lines.append(
            f"| `{entry['key']}` | {entry['name']} | {entry['category']} |"
            f" {entry['duration_s']:.0f} s | {'yes' if entry['demo'] else ''} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(markdown_report())
