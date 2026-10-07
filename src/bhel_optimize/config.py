"""Single source of truth for the project.

Holds the 30 DCS sensor tags, their boiler-part assignments, part
criticality weights, the steam-temperature anomaly rule, dataset paths and
train/val/test split fractions. Every other module (loader, preprocess,
simulation, detection, overview, UI) imports from here — do not duplicate
tag lists elsewhere.

Dataset quirk: the CSV headers carry a trailing ``#`` on every DCS tag
(e.g. ``PT_8313A.AV_0#``); use :func:`normalise_tag` instead of hard-coding
the suffix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "data.csv"  # as published (30 NaNs in YJJWSLL.AV_0)
IMPUTED_CSV = DATA_DIR / "data_AutoReg.csv"  # AutoReg-imputed, clean
COLUMNS_CSV = DATA_DIR / "columns.csv"  # tag descriptions (paper Table 1)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_PATH = ARTIFACTS_DIR / "scalers.joblib"

# --------------------------------------------------------------------------
# Dataset constants
# --------------------------------------------------------------------------

TAG_SUFFIX = "#"
DATE_COLUMN = "date"
SAMPLE_INTERVAL_S = 5

EXPECTED_ROWS = 86_400
EXPECTED_TAGS = 30

#: Boiler outlet steam temperature = the dataset's state-discriminating tag.
STATE_TAG = "TE_8332A.AV_0"
#: 530-545 degC is normal ("headed"); outside is anomalous ("tailed").
NORMAL_STEAM_TEMP_C = (530.0, 545.0)

#: Chronological split fractions (time-series aware, no shuffling).
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# --------------------------------------------------------------------------
# Tag descriptions (paper Table 1) — keyed by base tag name
# --------------------------------------------------------------------------

TAG_INFO: dict[str, str] = {
    "PT_8313A.AV_0": "Upper furnace pressure (A)",
    "PT_8313B.AV_0": "Upper furnace pressure (B)",
    "PT_8313C.AV_0": "Upper furnace pressure (C)",
    "PT_8313D.AV_0": "Upper furnace pressure (D)",
    "PT_8313E.AV_0": "Upper furnace pressure (E)",
    "PT_8313F.AV_0": "Upper furnace pressure (F)",
    "PTCA_8324.AV_0": "Container outlet vapour pressure",
    "PTCA_8322A.AV_0": "Pot pressure (left)",
    "TE_8319A.AV_0": "Flue gas temperature at the upper economizer outlet (left)",
    "TE_8319B.AV_0": "Flue gas temperature at the upper economizer outlet (right)",
    "TE_8313B.AV_0": "Temperature in the upper part of the furnace chamber (right)",
    "TE_8303.AV_0": "Primary air preheater outlet air temperature",
    "TE_8304.AV_0": "Secondary air preheater outlet air temperature",
    "TE_8332A.AV_0": "Boiler outlet steam temperature",
    "TV_8329ZC.AV_0": "Primary desuperheater outlet steam temperature regulating valve",
    "FT_8301.AV_0": "Primary fan outlet flow rate",
    "FT_8302.AV_0": "Secondary fan outlet flow",
    "FT_8306A.AV_0": "Return air chamber air flow (left)",
    "AIR_8301A.AV_0": "Upper economiser inlet flue gas oxygen (left)",
    "FT_8306B.AV_0": "Return air chamber air flow (right)",
    "AIR_8301B.AV_0": "Upper economiser inlet flue gas oxygen (right)",
    "YFJ3_AI.AV_0": "Induced draft fan motor current",
    "YFJ3_ZD1.AV_0": "Vibration of induced draft fan bearing shell (A)",
    "YFJ3_ZD2.AV_0": "Vibration of induced draft fan bearing shell (B)",
    "SXLTCYZ.AV_0": "Differential pressure between upper and lower hearth (left)",
    "SXLTCYY.AV_0": "Differential pressure between upper and lower hearth (right)",
    "ZCLCCY.AV_0": "Differential pressure in the left layer",
    "YCLCCY.AV_0": "Differential pressure in the right layer",
    "YJJWSLL.AV_0": "Primary desuperheating water flow output",
    "ZZQBCHLL.AV_0": "Main steam flow rate after compensation",
}

# --------------------------------------------------------------------------
# Boiler parts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    """A monitored boiler part and the tags that observe it."""

    name: str
    description: str
    criticality: int  # 1 (low) .. 5 (safety-critical)
    tags: tuple[str, ...]


PARTS: dict[str, Part] = {
    "furnace": Part(
        name="Furnace Chamber (Upper)",
        description="Combustion chamber; draft stability and bed behaviour",
        criticality=5,
        tags=(
            "PT_8313A.AV_0",
            "PT_8313B.AV_0",
            "PT_8313C.AV_0",
            "PT_8313D.AV_0",
            "PT_8313E.AV_0",
            "PT_8313F.AV_0",
            "TE_8313B.AV_0",
        ),
    ),
    "drum": Part(
        name="Steam Drum & Pot",
        description="Steam-water separation, drum pressure and level",
        criticality=5,
        tags=("PTCA_8322A.AV_0", "PTCA_8324.AV_0"),
    ),
    "main_steam": Part(
        name="Main Steam Line & Outlet",
        description="Steam outlet condition (state tag) and main steam flow",
        criticality=5,
        tags=("TE_8332A.AV_0", "ZZQBCHLL.AV_0"),
    ),
    "superheater": Part(
        name="Superheater & Desuperheater",
        description="Superheat temperature control via spray attemperation",
        criticality=4,
        tags=("TV_8329ZC.AV_0", "YJJWSLL.AV_0"),
    ),
    "economizer": Part(
        name="Economizer (Upper)",
        description="Flue-gas heat recovery; fouling and O2 trim",
        criticality=3,
        tags=(
            "TE_8319A.AV_0",
            "TE_8319B.AV_0",
            "AIR_8301A.AV_0",
            "AIR_8301B.AV_0",
        ),
    ),
    "air_preheaters": Part(
        name="Air Preheaters (Primary & Secondary)",
        description="Combustion air heating; fouling and leakage",
        criticality=3,
        tags=("TE_8303.AV_0", "TE_8304.AV_0"),
    ),
    "combustion_fans": Part(
        name="Combustion Air Fans (Primary & Secondary)",
        description="Primary/secondary air supply flows",
        criticality=3,
        tags=("FT_8301.AV_0", "FT_8302.AV_0"),
    ),
    "id_fan": Part(
        name="Induced Draft Fan",
        description="Flue-gas extraction; bearing wear and efficiency",
        criticality=4,
        tags=("YFJ3_AI.AV_0", "YFJ3_ZD1.AV_0", "YFJ3_ZD2.AV_0"),
    ),
    "return_air": Part(
        name="Return Air Chambers",
        description="Return air distribution (left/right)",
        criticality=2,
        tags=("FT_8306A.AV_0", "FT_8306B.AV_0"),
    ),
    "fluidized_bed": Part(
        name="Fluidized Bed (Hearth & Layers)",
        description="Bed fluidization and material circulation",
        criticality=4,
        tags=(
            "SXLTCYZ.AV_0",
            "SXLTCYY.AV_0",
            "ZCLCCY.AV_0",
            "YCLCCY.AV_0",
        ),
    ),
}

# --------------------------------------------------------------------------
# Derived lookups / helpers
# --------------------------------------------------------------------------

TAGS: tuple[str, ...] = tuple(TAG_INFO)
"""All 30 base tag names (header order of ``columns.csv``)."""

TAG_TO_PART: dict[str, str] = {
    tag: part_key for part_key, part in PARTS.items() for tag in part.tags
}


def normalise_tag(tag: str) -> str:
    """Strip the trailing ``#`` used in the CSV headers."""
    return tag[:-1] if tag.endswith(TAG_SUFFIX) else tag


def tags_for_part(part_key: str) -> tuple[str, ...]:
    """Return the tags monitoring ``part_key``."""
    return PARTS[part_key].tags


def part_of(tag: str) -> str:
    """Return the part key that monitors ``tag``."""
    return TAG_TO_PART[normalise_tag(tag)]


def criticality_of(tag: str) -> int:
    """Return the criticality weight (1-5) of the part monitoring ``tag``."""
    return PARTS[part_of(tag)].criticality


def part_keys() -> tuple[str, ...]:
    """Return all part keys in declaration order."""
    return tuple(PARTS)
