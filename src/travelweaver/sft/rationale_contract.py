"""Shared semantic contracts for visible SFT decision rationales."""

from __future__ import annotations

_PRICE_TERMS = ("价格", "标价", "价钱", "费用", "成本", "人均")
_COMPARISON_TERMS = ("高于", "低于", "更高", "更低", "更贵", "更便宜", "成本优势")


def has_visible_price_comparison(content: str) -> bool:
    """Return whether a removal rationale explains a price-based comparison.

    The teacher policy requires an immediately preceding candidate review and a
    cost-based removal rationale. Natural language may use ``标价`` rather than
    the template's literal ``价格``; the contract checks meaning rather than
    forcing a single surface form.
    """

    return (
        any(term in content for term in _PRICE_TERMS)
        and any(term in content for term in _COMPARISON_TERMS)
        and "移除" in content
    )
