"""
Tests for cssselect.optimize
============================

:copyright: (c) 2007-2012 Ian Bicking and contributors.
See AUTHORS for more details.
:license: BSD, see LICENSE for more details.

"""

from __future__ import annotations

import pytest
from lxml import etree, html

from cssselect import GenericTranslator, HTMLTranslator, optimize, optimize_parsed
from cssselect.parser import SelectorSyntaxError, parse
from cssselect.xpath import ExpressionError

HTML_DOC = html.fromstring(
    """
    <html><body>
      <div id="d1" class="foo bar" data-x="1">
        <p class="foo">one</p>
        <span class="bar"><a href="/x" class="foo link">link</a></span>
        <p class="baz">two</p>
      </div>
      <div id="d2" class="foo">
        <span class="foo bar">three</span>
      </div>
      <ul>
        <li class="item">1</li>
        <li class="item">2</li>
        <li class="item foo">3</li>
      </ul>
    </body></html>
    """
)

XML_DOC = etree.fromstring(
    """
    <root>
      <div id="d1" class="foo bar" data-x="1">
        <p class="foo">one</p>
        <span class="bar"><a href="/x" class="foo link">link</a></span>
        <p class="baz">two</p>
      </div>
      <div id="d2" class="foo">
        <span class="foo bar">three</span>
      </div>
      <ul>
        <li class="item">1</li>
        <li class="item">2</li>
        <li class="item foo">3</li>
      </ul>
    </root>
    """
)

REDUNDANT_SELECTORS = [
    # Duplicate conditions in a compound selector.
    "div.foo.foo",
    "#d1#d1",
    ".foo.bar.bar",
    "[data-x][data-x]",
    "p:first-child:first-child",
    "li:nth-child(2):nth-child(2)",
    "a[href][href].foo",
    ":not(.foo):not(.foo)",
    "div:has(> span):has(> span)",
    "ul li:nth-child(2n):nth-child(2n)",
    # Duplicate :is()/:where() branches.
    ":is(p, span, p)",
    "div > span:is(.bar, .bar)",
    "p:where(.foo, .foo)",
    # Absorption in :is()/:where().
    ":is(div, div.foo)",
    ":is(.foo, .foo.bar)",
    ":is(*, div)",
    # Always-true selector lists.
    "div:is(*)",
    "p:where(*)",
    # Single-branch lists.
    "div:is(div)",
    "*:is(span)",
    # Always-false compounds.
    "div:not(*)",
    "div.foo:not(div)",
    "div.foo:not(.foo)",
    "div:is(span)",
    "div:is(p, p)",
    # Redundancy inside combinators.
    "div p.foo.foo",
    "div > span:is(.bar, .bar)",
    "span > div.foo:not(div)",
]

# Selectors the optimizer must leave structurally alone: unknown
# extensions, unknown namespace bindings and complex relations are not
# guessed at.
UNTOUCHED_SELECTORS = [
    "div.foo",
    "div:has(span):has(span a)",
    "[foo='bar'][foo=bar]",
]

# Untranslatable selectors (unknown namespace prefix or pseudo-class):
# only their structure can be compared, not their hits.
UNTOUCHED_UNTRANSLATABLE_SELECTORS = [
    "ns|div:is(span)",
    "div:is(ns|span)",
    "div:unknown(1)",
]


def matched_elements(doc, xpath):
    tree = doc.getroottree()
    return {tree.getpath(element) for element in doc.xpath(xpath)}


def xpath_condition_count(xpath: str) -> int:
    return xpath.count(" and ") + xpath.count(" or ")


@pytest.mark.parametrize("css", REDUNDANT_SELECTORS + UNTOUCHED_SELECTORS)
def test_same_hits_on_html_and_xml(css: str) -> None:
    """Optimizing must not change which elements match, on either
    document kind."""
    [result] = optimize(css)
    for doc, translator in (
        (HTML_DOC, HTMLTranslator()),
        (XML_DOC, GenericTranslator()),
    ):
        before = matched_elements(doc, translator.css_to_xpath(css))
        after = matched_elements(doc, translator.selector_to_xpath(result.selector))
        assert before == after


@pytest.mark.parametrize("css", REDUNDANT_SELECTORS + UNTOUCHED_SELECTORS)
def test_specificity_is_preserved(css: str) -> None:
    """The optimized selector reports the specificity of the original,
    even when the simplified tree would compute a different one."""
    [result] = optimize(css)
    assert result.selector.specificity() == parse(css)[0].specificity()


def test_specificity_preserved_despite_simplification() -> None:
    [result] = optimize(".a.a")
    assert result.selector.canonical() == ".a"
    # The simplified tree alone would compute (0, 1, 0).
    assert result.selector.specificity() == (0, 2, 0)


@pytest.mark.parametrize("css", REDUNDANT_SELECTORS)
def test_optimized_xpath_has_fewer_conditions(css: str) -> None:
    """The generated XPath never gets longer or uses more conditions."""
    [result] = optimize(css)
    for translator in (GenericTranslator(), HTMLTranslator()):
        before = translator.css_to_xpath(css)
        after = translator.selector_to_xpath(result.selector)
        assert xpath_condition_count(after) <= xpath_condition_count(before)
        assert len(after) <= len(before)


@pytest.mark.parametrize(
    "css",
    [
        "div.foo.foo",
        "#d1#d1",
        ".foo.bar.bar",
        "[data-x][data-x]",
        "p:first-child:first-child",
        "li:nth-child(2):nth-child(2)",
        ":not(.foo):not(.foo)",
        ":is(p, span, p)",
        ":is(div, div.foo)",
        "div p.foo.foo",
    ],
)
def test_optimized_xpath_is_strictly_simpler(css: str) -> None:
    """Duplicate conditions and branches really do disappear from the
    generated XPath."""
    [result] = optimize(css)
    for translator in (GenericTranslator(), HTMLTranslator()):
        before = translator.css_to_xpath(css)
        after = translator.selector_to_xpath(result.selector)
        assert xpath_condition_count(after) < xpath_condition_count(before)


@pytest.mark.parametrize(
    "css", UNTOUCHED_SELECTORS + UNTOUCHED_UNTRANSLATABLE_SELECTORS
)
def test_unprovable_selectors_are_untouched(css: str) -> None:
    [result] = optimize(css)
    assert result.simplifications == []
    assert result.selector.canonical() == parse(css)[0].canonical()


@pytest.mark.parametrize(
    ("css", "rule", "canonical"),
    [
        ("div.foo.foo", "duplicate-condition", "div.foo"),
        ("p:is(a, a)", "duplicate-branch", ":not(*)"),
        (":is(p, span, p)", "duplicate-branch", ":is(p, span)"),
        (":is(div, div.foo)", "absorption", "div"),
        ("div:is(*)", "always-true", "div"),
        ("div:not(*)", "always-false", ":not(*)"),
        ("div:is(div)", "single-branch", "div"),
    ],
)
def test_simplification_records(css: str, rule: str, canonical: str) -> None:
    """Every applied simplification is recorded with its rule."""
    [result] = optimize(css)
    assert result.selector.canonical() == canonical
    assert rule in {s.rule for s in result.simplifications}
    for simplification in result.simplifications:
        assert simplification.before


def test_original_tree_is_not_modified() -> None:
    [selector] = parse("div.foo.foo")
    before = repr(selector.parsed_tree)
    result = optimize_parsed(selector)
    assert repr(selector.parsed_tree) == before
    assert selector.canonical() == "div.foo.foo"
    assert result.selector.canonical() == "div.foo"


def test_pseudo_element_is_preserved() -> None:
    [result] = optimize("a.foo.foo::before")
    assert result.selector.pseudo_element == "before"
    assert result.selector.specificity() == (0, 2, 2)
    assert result.selector.canonical() == "a.foo::before"


def test_selector_group() -> None:
    results = optimize("div.foo.foo, p:is(*)")
    assert len(results) == 2
    assert results[0].selector.canonical() == "div.foo"
    assert results[1].selector.canonical() == "p"


def test_invalid_input_raises_like_parse() -> None:
    """Invalid selectors raise through the regular parse entry point."""
    with pytest.raises(SelectorSyntaxError):
        optimize("div..foo")
    with pytest.raises(SelectorSyntaxError):
        optimize(":not(:not(.a))")
    with pytest.raises(SelectorSyntaxError):
        optimize("div:is()")


def test_unknown_pseudo_class_is_not_hidden() -> None:
    """The optimizer keeps unknown pseudo-classes, so translating the
    optimized selector still raises ExpressionError."""
    [result] = optimize("div.foo.foo:unknown(1)")
    assert ":unknown(1)" in result.selector.canonical()
    with pytest.raises(ExpressionError):
        GenericTranslator().selector_to_xpath(result.selector)


def test_zero_budget_leaves_tree_untouched() -> None:
    [selector] = parse("div.foo.foo")
    result = optimize_parsed(selector, budget=0)
    assert result.budget_exhausted
    assert result.simplifications == []
    assert result.selector.canonical() == "div.foo.foo"
    # The specificity is still the original one.
    assert result.selector.specificity() == (0, 2, 1)


def test_small_budget_still_equivalent() -> None:
    """A partial optimization under a tight budget stays equivalent."""
    css = "div.foo.foo > p.bar.bar:is(.baz, .baz)"
    [selector] = parse(css)
    result = optimize_parsed(selector, budget=3)
    assert result.budget_exhausted
    translator = GenericTranslator()
    before = matched_elements(XML_DOC, translator.css_to_xpath(css))
    after = matched_elements(XML_DOC, translator.selector_to_xpath(result.selector))
    assert before == after
    assert result.selector.specificity() == selector.specificity()


def test_deep_nesting_within_default_budget() -> None:
    css = "div" + ".foo" * 50 + ":is(*)"
    [result] = optimize(css)
    assert not result.budget_exhausted
    assert result.selector.canonical() == "div.foo"
