"""Tests for the boiler schematic: structure and geometry.

The diagram is drawn by hand in SVG, and the first version shipped a pile of
overlapping boxes and stray ducts. These tests parse the file and assert the
things a human reviewer would look for: every boiler part is drawn, each part
carries exactly one label and one live value, the process flow is present, and
no two part shapes overlap (except the fluidized bed, which is deliberately
drawn *inside* the furnace shell).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from bhel_optimize import config as C

SVG_PATH = C.PROJECT_ROOT / "templates" / "_boiler_svg.html"

#: The bed is drawn inside the furnace on purpose (it is a region of it).
ALLOWED_CONTAINMENT = {("fluidized_bed", "furnace")}
MIN_GAP = 1.0  # px, guards against boxes that merely touch


@pytest.fixture(scope="module")
def svg_root() -> ET.Element:
    # strip the Jinja-free HTML wrapper: the file is pure SVG markup
    text = SVG_PATH.read_text()
    return ET.fromstring(text)


def canvas_size(root: ET.Element) -> tuple[float, float]:
    """Width/height from the SVG's own viewBox (never hard-coded)."""
    parts = (root.get("viewBox") or "").replace(",", " ").split()
    if len(parts) != 4:
        raise AssertionError(f"unexpected viewBox: {root.get('viewBox')!r}")
    return float(parts[2]), float(parts[3])


def _tag(element: ET.Element) -> str:
    """Local tag name, ignoring any XML namespace."""
    return element.tag.split("}")[-1]


def _findall(element: ET.Element, tag: str) -> list[ET.Element]:
    """Namespace-agnostic descendant search (the file declares no xmlns)."""
    return [child for child in element.iter() if child is not element and _tag(child) == tag]


def _shape_box(element: ET.Element) -> tuple[float, float, float, float] | None:
    """Bounding box of a rect/circle/ellipse, else None."""
    kind = _tag(element)
    if kind == "rect":
        x, y = float(element.get("x", 0)), float(element.get("y", 0))
        return (x, y, x + float(element.get("width", 0)), y + float(element.get("height", 0)))
    if kind == "circle":
        cx, cy, r = (float(element.get(k, 0)) for k in ("cx", "cy", "r"))
        return (cx - r, cy - r, cx + r, cy + r)
    if kind == "ellipse":
        cx, cy = float(element.get("cx", 0)), float(element.get("cy", 0))
        rx, ry = float(element.get("rx", 0)), float(element.get("ry", 0))
        return (cx - rx, cy - ry, cx + rx, cy + ry)
    return None


def part_groups(root: ET.Element) -> dict[str, ET.Element]:
    return {
        group.get("data-part"): group
        for group in _findall(root, "g")
        if group.get("data-part") is not None
    }


def part_boxes(group: ET.Element) -> list[tuple[float, float, float, float]]:
    boxes = [_shape_box(child) for child in group]
    return [box for box in boxes if box is not None]


def part_shapes(root: ET.Element) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Every part shape with its owning part key.

    Geometry checks must use individual shapes, not each part's union box:
    several parts legitimately own *disjoint* pieces (the return-air chambers
    flank the furnace, the PA/SA fans are two circles), and a union box would
    span the gap between them and report phantom overlaps.
    """
    shapes: list[tuple[str, tuple[float, float, float, float]]] = []
    for key, group in part_groups(root).items():
        for box in part_boxes(group):
            shapes.append((key, box))
    return shapes


def path_end(d: str) -> tuple[float, float] | None:
    """End point of a path built from M/L/H/V/A commands (what this file uses).

    The arc command matters: the schematic draws a "hop" where one duct crosses
    another, so the parser must consume its seven parameters or it loses sync
    and reports a bogus end point.
    """
    tokens = re.findall(r"[MLHVAZmlhvaz]|-?\d+(?:\.\d+)?", d)
    x = y = 0.0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in "MmLl":
            x, y = float(tokens[index + 1]), float(tokens[index + 2])
            index += 3
        elif token in "Hh":
            x = float(tokens[index + 1])
            index += 2
        elif token in "Vv":
            y = float(tokens[index + 1])
            index += 2
        elif token in "Aa":
            # rx ry rotation large-arc-flag sweep-flag dx dy -> only dy/dx move
            x, y = x + float(tokens[index + 6]), y + float(tokens[index + 7])
            index += 8
        elif token in "Zz":
            index += 1
        else:  # bare coordinate pair continues the previous command
            x, y = float(tokens[index]), float(tokens[index + 1])
            index += 2
    return (x, y)


def overlaps(a, b, min_gap: float = MIN_GAP) -> bool:
    return not (
        a[2] + min_gap <= b[0]
        or b[2] + min_gap <= a[0]
        or a[3] + min_gap <= b[1]
        or b[3] + min_gap <= a[1]
    )


# --- structure ------------------------------------------------------------


def test_svg_parses_and_has_viewbox(svg_root):
    assert svg_root.get("id") == "boiler-svg"
    parts = (svg_root.get("viewBox") or "").split()
    assert len(parts) == 4, "the SVG must declare a viewBox"
    assert float(parts[0]) == 0 and float(parts[1]) == 0
    width, height = canvas_size(svg_root)
    assert width >= 600 and height >= 500, "canvas too small for a readable schematic"


def test_every_boiler_part_is_drawn(svg_root):
    assert set(part_groups(svg_root)) == set(C.PARTS)


def test_each_part_has_shapes_label_and_live_value(svg_root):
    """Each monitored box: a shape, a heading and exactly one live-value slot.

    The adopted design labels boxes with the ``th`` heading class (``ts`` for
    the tag lines), and the JS writes the live number into ``.part-value``.
    """
    for key, group in part_groups(svg_root).items():
        assert part_boxes(group), f"{key} has no shape"
        classes = [text.get("class", "") for text in _findall(group, "text")]
        assert any("th" in value.split() or "part-label" in value for value in classes), (
            f"{key} has no heading text"
        )
        assert sum("part-value" in value for value in classes) == 1, (
            f"{key} must have exactly one live-value slot"
        )


def test_every_part_is_keyboard_reachable(svg_root):
    for key, group in part_groups(svg_root).items():
        assert group.get("tabindex") == "0", f"{key} is not focusable"


def arrows(root: ET.Element) -> list[ET.Element]:
    """Ducts: <line> and <path> elements carrying the ``arr`` class."""
    return [
        node
        for tag in ("line", "path")
        for node in _findall(root, tag)
        if "arr" in (node.get("class") or "")
    ]


def test_process_flow_is_drawn(svg_root):
    """Steam, flue-gas, solids and air ducts must exist and point somewhere."""
    ducts = arrows(svg_root)
    assert len(ducts) >= 15, f"expected a full process flow, found {len(ducts)} ducts"
    strokes = " ".join(node.get("style", "") for node in ducts).upper()
    for circuit in ("#D85A30", "#1D9E75", "#7F77DD", "#888780"):
        assert circuit in strokes, f"missing {circuit} circuit ducts"
    with_markers = [node for node in ducts if node.get("marker-end")]
    assert len(with_markers) >= 15, "ducts must show flow direction"


def _text_anchor_x(node: ET.Element, width: float) -> tuple[float, float]:
    """Left/right extent of a text node given its anchor and estimated width."""
    x = float(node.get("x", 0))
    anchor = node.get("text-anchor", "start")
    if anchor == "middle":
        return x - width / 2, x + width / 2
    if anchor == "end":
        return x - width, x
    return x, x + width


def _est_width(node: ET.Element) -> float:
    """Rough advance width (browser metrics are close enough for a fit check)."""
    text = "".join(node.itertext())
    size = float(node.get("font-size", 14))
    mono = "part-value" in (node.get("class") or "")
    return len(text) * (0.60 if mono else 0.52) * size


def test_part_name_and_live_value_do_not_collide(svg_root):
    """The readout sits beside the part name, or on the line below it.

    Narrow boxes (the back-pass column) put the value on its own line under the
    tags; everything else shares the name's line. Either way the readout must
    stay inside the box and must not overlap any text it shares a line with.
    """
    for key, group in part_groups(svg_root).items():
        texts = _findall(group, "text")
        value = next(t for t in texts if "part-value" in (t.get("class") or ""))
        headings = [t for t in texts if "th" in (t.get("class") or "").split()]
        tags = [t for t in texts if "ts" in (t.get("class") or "").split()]
        assert headings, f"{key} has no name"
        box = part_boxes(group)[0]
        val_l, val_r = _text_anchor_x(value, _est_width(value))
        val_y = float(value.get("y"))

        # the return-air box is too narrow: its readout sits just outside
        if not (box[0] <= val_l and val_r <= box[2]):
            assert abs(val_l - box[2]) <= 24, f"{key}: detached value readout"
            continue

        # whichever line it is on, the value must not overlap that line's text
        same_line = [t for t in (*headings, *tags) if abs(float(t.get("y")) - val_y) <= 5]
        for node in same_line:
            _, other_r = _text_anchor_x(node, _est_width(node))
            other_l, _ = _text_anchor_x(node, _est_width(node))
            assert other_r + 3 <= val_l or val_r + 3 <= other_l, (
                f"{key}: value overlaps {''.join(node.itertext())!r}"
            )
        # on its own line it must still clear the neighbouring text vertically
        if not same_line:
            nearest = min(
                (abs(float(t.get("y")) - val_y) for t in (*headings, *tags)),
                default=99,
            )
            assert nearest >= 10, f"{key}: value crowds the line above it"


def test_svg_has_no_legend_block(svg_root):
    """The legend lives in the page's HTML, not inside the schematic.

    Checked against rendered text nodes (comments like
    ``<!-- coal feeder (not instrumented) -->`` are documentation, not content).
    """
    rendered = " ".join(
        " ".join(node.itertext()) for node in _findall(svg_root, "text")
    ).lower()
    for stale in ("not instrumented", "water / steam", "outline = current", "fluidized bed"):
        assert stale not in rendered, f"legend text still inside the SVG: {stale}"


def test_inactive_boxes_are_dark_and_injected_part_is_blue(svg_root):
    """Unmonitored boxes must be dark grey; an injected fault turns blue."""
    css = (C.PROJECT_ROOT / "static" / "css" / "app.css").read_text()
    gray_rule = css.split("#boiler-svg .c-gray > rect")[1].split("}")[0]
    assert "fill: #262c38" in gray_rule, "inactive boxes must be dark grey"
    fault_rule = css.split("#boiler-svg .part.fault > rect")[1].split("}")[0]
    assert "stroke: var(--accent)" in fault_rule, (
        "the injected fault must be outlined in the legend's blue accent"
    )
    # schematic is dark-only, so light text is always legible
    assert "--txt: #e8eef7" in css
    assert "background: #14161a" in css


def test_box_fill_follows_situation_level_in_css():
    """Boxes are coloured by level (the user's preference), pipes by circuit."""
    css = (C.PROJECT_ROOT / "static" / "css" / "app.css").read_text()
    for level in ("NOMINAL", "WATCH", "ALERT", "CRITICAL"):
        rule = f'#boiler-svg .part[data-level="{level}"] > rect'
        assert rule in css, f"missing level colour rule for {level}"
    assert "fill-opacity" in css and "fill: var(--nominal)" in css
    assert "#boiler-svg .part.fault" in css, "missing injected-fault styling"


def test_unmonitored_equipment_is_labelled(svg_root):
    """Plant that is not instrumented must still be drawn and named."""
    text = SVG_PATH.read_text().lower()
    for name in (
        "coal",
        "feeder",
        "cyclone",
        "refeeder",
        "dust collector",
        "stack",
        "turbine",
        "feedwater",
        "atmosphere",
        "hot air",
        "air to preheater",
    ):
        assert name in text, f"missing decorative label: {name}"


# --- geometry -------------------------------------------------------------


def test_part_shapes_do_not_overlap(svg_root):
    """No two part shapes may overlap (the diagram must be readable)."""
    shapes = part_shapes(svg_root)
    for index, (key_a, box_a) in enumerate(shapes):
        for key_b, box_b in shapes[index + 1:]:
            if key_a == key_b:
                continue
            if (key_a, key_b) in ALLOWED_CONTAINMENT or (key_b, key_a) in ALLOWED_CONTAINMENT:
                continue
            assert not overlaps(box_a, box_b), f"{key_a} {box_a} overlaps {key_b} {box_b}"


def test_part_shapes_stay_inside_the_canvas(svg_root):
    width, height = canvas_size(svg_root)
    for key, group in part_groups(svg_root).items():
        for box in part_boxes(group):
            assert box[0] >= 0 and box[1] >= 0, f"{key} starts outside the canvas"
            assert box[2] <= width and box[3] <= height, f"{key} exceeds the canvas"


def test_text_does_not_sit_on_another_part(svg_root):
    """Part labels/values must not land inside a different part's shape."""
    shapes = part_shapes(svg_root)
    for key, group in part_groups(svg_root).items():
        for text in _findall(group, "text"):
            x, y = float(text.get("x", 0)), float(text.get("y", 0))
            for other, box in shapes:
                if other == key or (key, other) in ALLOWED_CONTAINMENT:
                    continue
                inside = box[0] <= x <= box[2] and box[1] - 12 <= y <= box[3]
                assert not inside, f"{key} text at ({x},{y}) lands inside {other}"


def test_flow_paths_start_and_end_off_the_text_boxes(svg_root):
    """Duct endpoints must not terminate inside an unrelated part shape."""
    shapes = part_shapes(svg_root)
    for node in arrows(svg_root):
        classes = node.get("class") or ""
        # "inner" arrows are drawn *inside* a vessel on purpose (the combustion
        # gases rising through the furnace), so they are not ducts to a box
        if "inner" in classes:
            continue
        if _tag(node) == "line":
            end = (float(node.get("x2", 0)), float(node.get("y2", 0)))
        else:
            end = path_end(node.get("d", ""))
        if end is None:
            continue
        for key, box in shapes:
            if box[0] < end[0] < box[2] and box[1] < end[1] < box[3]:
                pytest.fail(f"duct ends inside {key} at {end}")
