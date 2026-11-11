"""
Tests for the optional selector optimizer (cssselect.optimize).

The optimizer is checked against small, hand-enumerated HTML/XML trees:
optimized and original selectors must hit exactly the same elements and
report the same specificity, while the generated XPath must carry no more
conditions than the original.

:copyright: (c) 2007-2012 Ian Bicking and contributors.
See AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import typing

import pytest
from lxml import etree

from cssselect import (
    GenericTranslator,
    Selector,
    SelectorSyntaxError,
    optimize_selector,
    parse,
)
from cssselect.optimize import (
    BUDGET_EXCEEDED,
    OptimizationRecord,
    _OptimizedSelector,
)

# An enumerated HTML tree: every interesting combination of element type,
# id, class, attribute and nesting appears at least once.
HTML_TREE = """
<html>
  <body>
    <div id="d1" class="a b" lang="en">
      <p id="p1" class="a c">text</p>
      <p id="p2" class="a a">
        <span id="s1" class="b">deep</span>
      </p>
    </div>
    <div id="d2" class="a c" lang="fr">
      <p id="p3" class="b c"/>
    </div>
    <div id="d3">
      <span id="s2" class="a c"/>
    </div>
    <section id="d4" class="a b"/>
    <ul id="u1">
      <li id="l1" class="x"/>
      <li id="l2" class="x"/>
    </ul>
  </body>
</html>
"""

# An XML tree exercising namespaces.
XML_TREE = """
<root xmlns="http://example.org/ns" xmlns:other="http://example.org/other">
  <div id="x1" class="a"/>
  <other:div id="x2" other:class="a"/>
</root>
"""

# Selectors that must be simplified, with the record kind each rewrite
# produces.
REWRITE_CASES: list[tuple[str, str]] = [
    ("div.a.a", "duplicate-condition"),
    ("div#d1#d1", "duplicate-condition"),
    ("div[lang='en'][lang='en']", "duplicate-condition"),
    ("div.a.a.a", "duplicate-condition"),
    ("div > p.a.a", "duplicate-condition"),
    ("div.a:is(.a, .b, .a)", "duplicate-condition"),
    ("div.a:where(.a, .b, .a)", "duplicate-condition"),
    ("div.a:has(.b, .b)", "duplicate-condition"),
    ("div:is(*)", "always-true"),
    ("div.a:is(div.a)", "always-true"),
    ("div.a:where(div.a)", "always-true"),
    ("div.a:not(div.a)", "always-false"),
    ("div.a:not(.a)", "always-false"),
    ("div.a.b:not(div.a)", "always-false"),
    ("div.a.b:not(.a)", "always-false"),
    ("div.a:is(div.a.b)", "absorption"),
]

# Selectors the optimizer must leave alone: it cannot prove the rewrite
# equivalent.
UNTOUCHED_CASES = [
    "div.a:is(div.a.b, .a)",
    "div.a:is(div.a.b, div.a.c, .z)",
    "div.a:not(div.a.b)",
    "div.a.b:not(div.c)",
    "div:not(span)",
    "div:has(> .b)",
    "div:scope",
    "li:nth-child(2n):nth-child(2n)",
    "div.a:has(> .b, > .c)",
    "span.a:is(div.a, .b)",
]


def _condition_count(xpath: str) -> int:
    """Count the conditions in an XPath expression: predicates plus the
    boolean operators joining them."""
    return xpath.count("[") + xpath.count(" and ") + xpath.count(" or ")


@pytest.fixture(scope="module")
def document() -> etree._Element:
    return etree.fromstring(HTML_TREE)


@pytest.fixture(scope="module")
def translator() -> GenericTranslator:
    return GenericTranslator()


def _hits(
    translator: GenericTranslator, document: etree._Element, css: str
) -> list[str]:
    xpath = translator.css_to_xpath(css)
    elements = typing.cast("list[etree._Element]", document.xpath(xpath))
    return sorted(element.get("id", "nil") for element in elements)


def _xpath_for(translator: GenericTranslator, selector: Selector) -> str:
    return translator.selector_to_xpath(selector)


class TestOptimizeSelector:
    @pytest.mark.parametrize(("css", "kind"), REWRITE_CASES)
    def test_rewrite_applies(
        self, translator: GenericTranslator, css: str, kind: str
    ) -> None:
        selector = parse(css)[0]
        result = optimize_selector(selector)
        assert kind in [record.kind for record in result.records]
        assert result.selector.canonical() != selector.canonical()

    @pytest.mark.parametrize("css", UNTOUCHED_CASES)
    def test_unprovable_left_alone(self, css: str) -> None:
        selector = parse(css)[0]
        result = optimize_selector(selector)
        assert result.records == ()
        assert result.selector.canonical() == selector.canonical()

    @pytest.mark.parametrize(("css", "kind"), REWRITE_CASES)
    @pytest.mark.parametrize("untouched", UNTOUCHED_CASES)
    def test_hits_and_specificity_preserved(
        self,
        translator: GenericTranslator,
        document: etree._Element,
        css: str,
        kind: str,
        untouched: str,
    ) -> None:
        # Cross-product: every case, rewritten or not, must select exactly
        # the same elements before and after optimization.
        for candidate in (css, untouched):
            selector = parse(candidate)[0]
            result = optimize_selector(selector)
            before = _hits(translator, document, candidate)
            after_xpath = _xpath_for(translator, result.selector)
            elements = typing.cast(
                "list[etree._Element]", document.xpath(after_xpath)
            )
            after = sorted(element.get("id", "nil") for element in elements)
            assert after == before
            assert result.selector.specificity() == selector.specificity()

    @pytest.mark.parametrize(("css", "kind"), REWRITE_CASES)
    def test_xpath_not_longer(
        self, translator: GenericTranslator, css: str, kind: str
    ) -> None:
        selector = parse(css)[0]
        result = optimize_selector(selector)
        before = _condition_count(_xpath_for(translator, selector))
        after = _condition_count(_xpath_for(translator, result.selector))
        assert after <= before

    def test_condition_count_decreases(
        self, translator: GenericTranslator
    ) -> None:
        # The motivating case: duplicated conditions and :is() branches
        # must disappear from the generated XPath.
        selector = parse("div[lang='en'][lang='en']")[0]
        result = optimize_selector(selector)
        before = _condition_count(_xpath_for(translator, selector))
        after = _condition_count(_xpath_for(translator, result.selector))
        assert after < before
        # A class predicate is a single bracket, but the duplicate still
        # collapses the two @class tests inside it.
        class_selector = parse("div.a.a")[0]
        class_result = optimize_selector(class_selector)
        assert len(class_result.selector.canonical()) < len(
            class_selector.canonical()
        )

    def test_original_tree_not_modified(self) -> None:
        selector = parse("div.a.a:is(.a, .b, .a)")[0]
        snapshot = repr(selector)
        optimize_selector(selector)
        assert repr(selector) == snapshot

    def test_idempotent(self) -> None:
        for css, _ in REWRITE_CASES:
            result = optimize_selector(parse(css)[0])
            again = optimize_selector(result.selector)
            assert again.selector.canonical() == result.selector.canonical()

    def test_records_are_frozen_dataclasses(self) -> None:
        result = optimize_selector(parse("div.a.a")[0])
        (record,) = result.records
        assert isinstance(record, OptimizationRecord)
        assert record.kind == "duplicate-condition"
        assert record.detail
        with pytest.raises(AttributeError):
            record.kind = "other"  # type: ignore[misc]

    def test_specificity_pinned_not_recomputed(self) -> None:
        # "div.a.a" is (0, 2, 1); the simplified "div.a" would be (0, 1, 1)
        # if specificity were recomputed from the new tree.
        selector = parse("div.a.a")[0]
        result = optimize_selector(selector)
        assert result.selector.canonical() == "div.a"
        assert result.selector.specificity() == (0, 2, 1)
        assert isinstance(result.selector, _OptimizedSelector)

    def test_expansion_budget(self) -> None:
        # The absorption of "div.a:is(div.a.b, div.b)" emits two list
        # members, so a budget of 1 leaves the selector untouched...
        selector = parse("div.a:is(div.a.b, div.b)")[0]
        result = optimize_selector(selector, max_expansions=1)
        assert result.records[-1].kind == BUDGET_EXCEEDED
        assert result.selector.canonical() == selector.canonical()
        # ...while the default budget simplifies it.
        result = optimize_selector(selector)
        assert result.selector.canonical() == "div.a.b"

    def test_invalid_budget(self) -> None:
        with pytest.raises(ValueError, match="max_expansions"):
            optimize_selector(parse("div")[0], max_expansions=0)

    def test_namespaced_xml_tree_unchanged(self) -> None:
        document = etree.fromstring(XML_TREE)
        translator = GenericTranslator()
        namespaces = {"other": "http://example.org/other"}
        for css in ("div.a.a", "other|div.a.a"):
            selector = parse(css)[0]
            result = optimize_selector(selector)
            before = translator.css_to_xpath(css)
            after = _xpath_for(translator, result.selector)
            before_hits = document.xpath(before, namespaces=namespaces)
            after_hits = document.xpath(after, namespaces=namespaces)
            assert before_hits == after_hits

    def test_namespaces_are_not_guessed_with(self) -> None:
        # Namespaced duplicates and namespace-prefixed attributes are never
        # deduplicated: the optimizer cannot prove the bindings equal.
        for css in (
            "other|div.a.a",
            "div[other|lang='en'][other|lang='en']",
            "div[other|lang='en'][lang='en']",
        ):
            selector = parse(css)[0]
            result = optimize_selector(selector)
            assert result.records == ()
            assert result.selector.canonical() == selector.canonical()

    def test_errors_not_hidden(self) -> None:
        # Invalid input still fails at the usual entry point; the optimizer
        # never parses, so it cannot mask unsupported syntax.
        with pytest.raises(SelectorSyntaxError):
            parse("div..a")
        with pytest.raises(SelectorSyntaxError):
            parse("div:a(")
        # An unsupported pseudo-class still raises at translation time,
        # optimized or not.
        from cssselect import ExpressionError

        selector = parse("div:unknown-pseudo")[0]
        result = optimize_selector(selector)
        translator = GenericTranslator()
        with pytest.raises(ExpressionError):
            translator.selector_to_xpath(selector)
        with pytest.raises(ExpressionError):
            translator.selector_to_xpath(result.selector)

    def test_pseudo_element_preserved(self) -> None:
        selector = parse("div.a.a::before")[0]
        result = optimize_selector(selector)
        assert result.selector.pseudo_element == "before"
        assert result.selector.specificity() == selector.specificity()

    def test_selector_group(self) -> None:
        selectors = parse("div.a.a, span.b.b")
        canonical = []
        for selector in selectors:
            result = optimize_selector(selector)
            assert result.selector.specificity() == selector.specificity()
            canonical.append(result.selector.canonical())
        assert canonical == ["div.a", "span.b"]

    def test_pinned_specificity_not_recomputed_on_reparse(self) -> None:
        selector = parse("div.a.a")[0]
        result = optimize_selector(selector)
        assert result.selector.specificity() == (0, 2, 1)
        # The serialized form is the simplified selector; re-parsing it
        # yields the simplified specificity, not the pinned one.
        reparsed = parse(result.selector.canonical())[0]
        assert reparsed.canonical() == "div.a"
        assert reparsed.specificity() == (0, 1, 1)

    def test_functional_pseudo_left_alone(
        self, translator: GenericTranslator, document: etree._Element
    ) -> None:
        # :nth-child() and friends are never rewritten.
        selector = parse("li:nth-child(2n):nth-child(2n)")[0]
        result = optimize_selector(selector)
        assert result.records == ()
        assert _hits(translator, document, "li:nth-child(2n):nth-child(2n)") == sorted(
            element.get("id", "nil")
            for element in document.xpath(
                _xpath_for(translator, result.selector)
            )
        )
