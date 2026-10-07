"""Tests for the config part/tag map — the single source of truth."""

from __future__ import annotations

from bhel_optimize import config as C


def test_tag_and_part_counts():
    assert len(C.TAGS) == C.EXPECTED_TAGS == 30
    assert len(C.PARTS) == 10


def test_tag_info_covers_all_tags():
    assert set(C.TAG_INFO) == set(C.TAGS)


def test_every_tag_assigned_to_exactly_one_part():
    assigned = [tag for part in C.PARTS.values() for tag in part.tags]
    assert len(assigned) == len(C.TAGS)
    assert len(set(assigned)) == len(assigned), "a tag is mapped to two parts"
    assert set(assigned) == set(C.TAGS)


def test_tag_to_part_lookup_is_complete():
    assert set(C.TAG_TO_PART) == set(C.TAGS)


def test_criticality_weights_valid():
    for part in C.PARTS.values():
        assert 1 <= part.criticality <= 5, part.name


def test_helpers_are_consistent():
    for part_key, part in C.PARTS.items():
        assert C.tags_for_part(part_key) == part.tags
        for tag in part.tags:
            assert C.part_of(tag) == part_key
            assert C.criticality_of(tag) == part.criticality


def test_state_tag_is_in_main_steam_part():
    assert C.STATE_TAG in C.TAGS
    assert C.part_of(C.STATE_TAG) == "main_steam"


def test_normal_steam_temp_range_is_sane():
    low, high = C.NORMAL_STEAM_TEMP_C
    assert low < high
    assert (low, high) == (530.0, 545.0)


def test_split_fractions_sum_to_one():
    assert abs(sum(C.SPLIT_FRACTIONS.values()) - 1.0) < 1e-9


def test_normalise_tag_strips_suffix():
    assert C.normalise_tag("PT_8313A.AV_0#") == "PT_8313A.AV_0"
    assert C.normalise_tag("PT_8313A.AV_0") == "PT_8313A.AV_0"


def test_part_keys_are_stable():
    assert set(C.part_keys()) == set(C.PARTS)
