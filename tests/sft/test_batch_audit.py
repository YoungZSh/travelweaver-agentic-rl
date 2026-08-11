from travelweaver.sft.batch_audit import (
    _catalog_selection_is_visible,
    _distribution,
    _duration,
)
from travelweaver.sft.rationale_contract import has_visible_price_comparison


def test_batch_audit_helpers_handle_percentiles_and_midnight() -> None:
    assert _duration("23:50", "00:20") == 30
    assert _duration("10:00", "10:45") == 45
    assert _distribution([1, 2, 3, 4, 5]) == {
        "min": 1,
        "mean": 3,
        "p50": 3,
        "p90": 5,
        "max": 5,
    }


def test_catalog_grounding_accepts_compound_raw_facets() -> None:
    visible = {"博物馆", "纪念馆", "公园"}

    assert _catalog_selection_is_visible("博物馆/纪念馆", visible)
    assert not _catalog_selection_is_visible("博物馆/艺术馆", visible)


def test_remove_rationale_accepts_natural_price_comparison_synonyms() -> None:
    assert has_visible_price_comparison(
        "清单里甲标价199元，而乙是0元，前者没有成本优势，所以把它移除。"
    )
    assert has_visible_price_comparison("甲人均更贵，因此移除这个候选。")
    assert not has_visible_price_comparison("甲看起来不合适，因此移除这个候选。")
