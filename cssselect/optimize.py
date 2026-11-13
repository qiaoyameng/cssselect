"""
cssselect.optimize
==================

An optional optimization pass for parsed selectors.

The pass rewrites a parsed tree into a smaller, provably equivalent one
and records every simplification it applies.  The original tree is
never modified.  Only rewrites that are equivalent regardless of the
document and of the semantics of unknown pseudo-classes are performed:

* duplicate conditions in a compound selector (``.a.a`` -> ``.a``),
* duplicate branches in ``:is()``/``:where()``/``:has()`` lists,
* absorption in ``:is()``/``:where()`` (``:is(div, div.x)`` -> ``div``),
* always-true selector lists (``div:is(*)`` -> ``div``) and
* always-false compounds (``div:not(div)``, ``div:is(span)``).

Unknown extensions, differing namespace bindings and complex relations
are left untouched.  The specificity of the original selector is
preserved on the result, even when the simplified tree would compute a
different one.

:copyright: (c) 2007-2012 Ian Bicking and contributors.
See AUTHORS for more details.
:license: BSD, see LICENSE for more details.

"""

from __future__ import annotations

from dataclasses import dataclass, field

from cssselect.parser import (
    Attrib,
    Class,
    CombinedSelector,
    Element,
    Function,
    Hash,
    Matching,
    Negation,
    Pseudo,
    PseudoElement,
    Relation,
    Selector,
    SpecificityAdjustment,
    Token,
    Tree,
    parse,
)

#: Default maximum number of tree nodes the optimizer visits before
#: giving up, so that optimizing cannot cause an exponential amount of
#: intermediate work.
DEFAULT_BUDGET = 1000

#: Structural key for a condition or a whole tree.  Only exact
#: structural equality is ever relied upon.
_Key = tuple


@dataclass
class Simplification:
    """One simplification applied by the optimizer.

    ``rule`` is one of ``"duplicate-condition"``, ``"duplicate-branch"``,
    ``"absorption"``, ``"single-branch"``, ``"always-true"`` or
    ``"always-false"``.  ``before`` and ``after`` are canonical CSS
    representations of the removed or replaced fragment (``after`` is
    empty for removals).
    """

    rule: str
    before: str
    after: str = ""


@dataclass
class OptimizationResult:
    """The outcome of optimizing a single parsed selector."""

    #: The optimized selector.  Its ``specificity()`` is the one of the
    #: original selector, even when the simplified tree would compute a
    #: different one.
    selector: Selector
    #: The simplifications that were applied, in application order.
    simplifications: list[Simplification] = field(default_factory=list)
    #: Whether the expansion budget ran out before the whole tree could
    #: be optimized.  The result is still equivalent, only less
    #: simplified.
    budget_exhausted: bool = False


class OptimizedSelector(Selector):
    """A :class:`Selector` whose specificity is taken from the selector
    it was optimized from, not recomputed from the simplified tree."""

    def __init__(
        self,
        tree: Tree,
        pseudo_element: PseudoElement | None,
        original_specificity: tuple[int, int, int],
    ) -> None:
        super().__init__(tree, pseudo_element)
        self._original_specificity = original_specificity

    def specificity(self) -> tuple[int, int, int]:
        return self._original_specificity


def optimize(css: str, *, budget: int = DEFAULT_BUDGET) -> list[OptimizationResult]:
    """Parse and optimize a CSS *group of selectors*.

    Parsing goes through the regular :func:`~cssselect.parser.parse`
    entry point, so invalid selectors raise
    :class:`~cssselect.parser.SelectorSyntaxError` exactly as before;
    the optimizer never accepts or hides unsupported syntax.

    :param budget:
        Maximum number of tree nodes visited while optimizing; see
        :data:`DEFAULT_BUDGET`.
    :returns:
        A list of :class:`OptimizationResult`, one for each selector in
        the comma-separated group.

    """
    return [optimize_parsed(selector, budget=budget) for selector in parse(css)]


def optimize_parsed(
    selector: Selector, *, budget: int = DEFAULT_BUDGET
) -> OptimizationResult:
    """Optimize an already-parsed :class:`Selector`.

    The selector's own tree is not modified; the result may share
    unchanged subtrees with it.
    """
    optimizer = _Optimizer(budget)
    new_tree = optimizer.optimize_tree(selector.parsed_tree)
    optimized = OptimizedSelector(
        new_tree, selector.pseudo_element, selector.specificity()
    )
    return OptimizationResult(optimized, optimizer.records, optimizer.exhausted)


#### Structural keys


def _token_key(token: Token) -> tuple[str, str | None]:
    return (token.type, token.value)


def _condition_key(condition: Tree) -> _Key:
    """A key for a compound-selector condition, ignoring the selector it
    is applied to (its position in the compound chain)."""
    if isinstance(condition, Hash):
        return ("hash", condition.id)
    if isinstance(condition, Class):
        return ("class", condition.class_name)
    if isinstance(condition, Pseudo):
        return ("pseudo", condition.ident)
    if isinstance(condition, Function):
        return (
            "function",
            condition.name,
            tuple(_token_key(token) for token in condition.arguments),
        )
    if isinstance(condition, Attrib):
        value = None if condition.value is None else _token_key(condition.value)
        return (
            "attrib",
            condition.namespace,
            condition.attrib,
            condition.operator,
            value,
            condition.flag,
        )
    if isinstance(condition, Negation):
        return ("not", _tree_key(condition.subselector))
    if isinstance(condition, Matching):
        return ("is", tuple(_tree_key(s) for s in condition.selector_list))
    if isinstance(condition, SpecificityAdjustment):
        return ("where", tuple(_tree_key(s) for s in condition.selector_list))
    if isinstance(condition, Relation):
        return (
            "has",
            tuple(
                (combinator.value, _tree_key(subselector.parsed_tree))
                for combinator, subselector in condition.arguments
            ),
        )
    raise TypeError(f"Not a compound-selector condition: {condition!r}")


def _tree_key(tree: Tree) -> _Key:
    if isinstance(tree, Element):
        return ("element", tree.namespace, tree.element)
    if isinstance(tree, CombinedSelector):
        return (
            "combined",
            _tree_key(tree.selector),
            tree.combinator,
            _tree_key(tree.subselector),
        )
    selector = tree.selector  # type: ignore[attr-defined]
    return ("compound", _tree_key(selector), _condition_key(tree))


#### Compound selectors


def _split_compound(tree: Tree) -> tuple[Element, list[Tree]] | None:
    """Split a pure compound selector into its base element test and its
    conditions, in source order.  Return None for complex selectors."""
    conditions = []
    node = tree
    while not isinstance(node, Element):
        if isinstance(node, CombinedSelector):
            return None
        conditions.append(node)
        node = node.selector  # type: ignore[attr-defined]
    conditions.reverse()
    return node, conditions


def _compound_condition_set(tree: Tree) -> frozenset[_Key] | None:
    """The set of predicates a pure compound selector ANDs together,
    including its element test, or None for complex selectors."""
    split = _split_compound(tree)
    if split is None:
        return None
    base, conditions = split
    keys = {_condition_key(condition) for condition in conditions}
    if not _is_universal(base):
        keys.add(_tree_key(base))
    return frozenset(keys)


def _is_universal(element: Element) -> bool:
    """Whether this element test matches every element.  Only the
    universal selector and the explicit any-namespace wildcard are
    considered; anything with a namespace prefix is not guessed at."""
    return element.element is None and element.namespace in (None, "*")


def _rebuild_condition(condition: Tree, selector: Tree) -> Tree:
    """Copy a condition node, applying it to a new inner selector."""
    if isinstance(condition, Hash):
        return Hash(selector, condition.id)
    if isinstance(condition, Class):
        return Class(selector, condition.class_name)
    if isinstance(condition, Pseudo):
        return Pseudo(selector, condition.ident)
    if isinstance(condition, Function):
        return Function(selector, condition.name, condition.arguments)
    if isinstance(condition, Attrib):
        return Attrib(
            selector,
            condition.namespace,
            condition.attrib,
            condition.operator,  # type: ignore[arg-type]
            condition.value,  # type: ignore[arg-type]
            condition.flag,
        )
    if isinstance(condition, Negation):
        return Negation(selector, condition.subselector)
    if isinstance(condition, Matching):
        return Matching(selector, condition.selector_list)
    if isinstance(condition, SpecificityAdjustment):
        return SpecificityAdjustment(selector, condition.selector_list)
    if isinstance(condition, Relation):
        return Relation(selector, condition.arguments)
    raise TypeError(f"Not a compound-selector condition: {condition!r}")


class _Drop:
    """Sentinel: this condition is always true and can be removed."""


class _BaseConflict:
    """Sentinel: merging this branch makes the compound unsatisfiable."""


class _Merge:
    """Sentinel: this selector-list condition holds a single compound
    branch whose conditions can be merged into the enclosing compound.
    ``fallback`` is the condition to keep when the merge cannot be
    proven safe."""

    def __init__(self, branch: Tree, fallback: Tree) -> None:
        self.branch = branch
        self.fallback = fallback


#: The canonical always-false compound: ``*:not(*)``.
_ALWAYS_FALSE = Negation(Element(), Element())


class _Optimizer:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.records: list[Simplification] = []
        self.exhausted = False

    def spend(self) -> bool:
        if self.budget <= 0:
            self.exhausted = True
            return False
        self.budget -= 1
        return True

    def optimize_tree(self, tree: Tree) -> Tree:
        if not self.spend():
            return tree
        if isinstance(tree, CombinedSelector):
            selector = self.optimize_tree(tree.selector)
            subselector = self.optimize_tree(tree.subselector)
            if selector is tree.selector and subselector is tree.subselector:
                return tree
            return CombinedSelector(selector, tree.combinator, subselector)
        if isinstance(tree, Element):
            return tree
        return self.optimize_compound(tree)

    def optimize_compound(self, tree: Tree) -> Tree:
        split = _split_compound(tree)
        assert split is not None
        base, conditions = split
        new_conditions: list[Tree] = []
        changed = False
        for condition in conditions:
            if not self.spend():
                new_conditions.append(condition)
                continue
            optimized = self.optimize_condition(condition)
            if isinstance(optimized, _Drop):
                changed = True
                continue
            if isinstance(optimized, _Merge):
                branch_split = _split_compound(optimized.branch)
                assert branch_split is not None
                branch_base, branch_conditions = branch_split
                combined = _combine_bases(base, branch_base)
                if isinstance(combined, _BaseConflict):
                    self.records.append(
                        Simplification(
                            "always-false",
                            tree.canonical(),
                            _ALWAYS_FALSE.canonical(),
                        )
                    )
                    return _ALWAYS_FALSE
                if combined is None:
                    # Namespace bindings differ or are unknown: do not
                    # guess, keep the single-branch list as a condition.
                    new_conditions.append(optimized.fallback)
                    if optimized.fallback is not condition:
                        changed = True
                    continue
                self.records.append(
                    Simplification(
                        "single-branch",
                        optimized.fallback.canonical(),
                        optimized.branch.canonical(),
                    )
                )
                base = combined
                new_conditions.extend(branch_conditions)
                changed = True
                continue
            new_conditions.append(optimized)
            if optimized is not condition:
                changed = True
        if self.spend():
            new_conditions, changed = self._dedup_conditions(new_conditions, changed)
            false = self._check_always_false(tree, base, new_conditions)
            if false is not None:
                return false
        if not changed:
            return tree
        node: Tree = base
        for condition in new_conditions:
            node = _rebuild_condition(condition, node)
        return node

    def _dedup_conditions(
        self, conditions: list[Tree], changed: bool
    ) -> tuple[list[Tree], bool]:
        # Idempotence of "and": drop duplicate conditions, keeping the
        # first occurrence.
        seen: set[_Key] = set()
        deduped: list[Tree] = []
        for condition in conditions:
            key = _condition_key(condition)
            if key in seen:
                self.records.append(
                    Simplification("duplicate-condition", condition.canonical())
                )
                changed = True
            else:
                seen.add(key)
                deduped.append(condition)
        return deduped, changed

    def _check_always_false(
        self, tree: Tree, base: Element, conditions: list[Tree]
    ) -> Tree | None:
        # A compound "X:not(Y)" is unsatisfiable when every predicate of
        # Y is also a predicate of X.
        available = {_condition_key(condition) for condition in conditions}
        for condition in conditions:
            if not isinstance(condition, Negation):
                continue
            sub_split = _split_compound(condition.subselector)
            if sub_split is None:
                continue
            sub_base, sub_conditions = sub_split
            if not (_is_universal(sub_base) or _tree_key(sub_base) == _tree_key(base)):
                continue
            others = available - {_condition_key(condition)}
            if all(_condition_key(c) in others for c in sub_conditions):
                self.records.append(
                    Simplification(
                        "always-false", tree.canonical(), _ALWAYS_FALSE.canonical()
                    )
                )
                return _ALWAYS_FALSE
        return None

    def optimize_condition(self, condition: Tree) -> Tree | _Drop | _Merge:
        if isinstance(condition, Negation):
            subselector = self.optimize_tree(condition.subselector)
            if subselector is condition.subselector:
                return condition
            return Negation(condition.selector, subselector)
        if isinstance(condition, (Matching, SpecificityAdjustment)):
            return self.optimize_selector_list(condition)
        if isinstance(condition, Relation):
            return self.optimize_relation(condition)
        return condition

    def optimize_selector_list(
        self, condition: Matching | SpecificityAdjustment
    ) -> Tree | _Drop | _Merge:
        branches = []
        changed = False
        for branch in condition.selector_list:
            optimized = self.optimize_tree(branch)
            branches.append(optimized)
            if optimized is not branch:
                changed = True
        # Idempotence of "or": drop duplicate branches, keeping the
        # first occurrence.
        seen: set[_Key] = set()
        deduped = []
        for branch in branches:
            key = _tree_key(branch)
            if key in seen:
                self.records.append(
                    Simplification("duplicate-branch", branch.canonical())
                )
                changed = True
            else:
                seen.add(key)
                deduped.append(branch)
        # Absorption: a branch whose predicate set is a strict superset
        # of another branch's matches a subset of elements and is
        # redundant in an "or" list.
        condition_sets = [_compound_condition_set(branch) for branch in deduped]
        kept = []
        for i, branch in enumerate(deduped):
            branch_set = condition_sets[i]
            if branch_set is not None and any(
                j != i and other is not None and other < branch_set
                for j, other in enumerate(condition_sets)
            ):
                self.records.append(Simplification("absorption", branch.canonical()))
                changed = True
            else:
                kept.append(branch)
        kept_split = _split_compound(kept[0]) if len(kept) == 1 else None
        if kept_split is not None:
            branch_base, branch_conditions = kept_split
            if _is_universal(branch_base) and not branch_conditions:
                # ":is(*)" / ":where(*)" match every element.
                self.records.append(
                    Simplification("always-true", condition.canonical())
                )
                return _Drop()
            fallback = condition if not changed else type(condition)(
                condition.selector, kept
            )
            return _Merge(kept[0], fallback)
        if not changed:
            return condition
        return type(condition)(condition.selector, kept)

    def optimize_relation(self, condition: Relation) -> Tree:
        arguments = []
        changed = False
        seen: set[_Key] = set()
        for combinator, subselector in condition.arguments:
            new_tree = self.optimize_tree(subselector.parsed_tree)
            if new_tree is not subselector.parsed_tree:
                subselector = Selector(new_tree, subselector.pseudo_element)
                changed = True
            key = (combinator.value, _tree_key(subselector.parsed_tree))
            if key in seen:
                self.records.append(
                    Simplification(
                        "duplicate-branch", subselector.parsed_tree.canonical()
                    )
                )
                changed = True
                continue
            seen.add(key)
            arguments.append((combinator, subselector))
        if not changed:
            return condition
        return Relation(condition.selector, arguments)


def _combine_bases(
    base: Element, branch_base: Element
) -> Element | _BaseConflict | None:
    """Combine the element tests of a compound and of a single-branch
    selector list being merged into it.  Return None when equivalence
    cannot be proven (different namespace bindings)."""
    if _is_universal(branch_base):
        return base
    if _is_universal(base):
        return branch_base
    if base.namespace == branch_base.namespace:
        if base.element == branch_base.element:
            return base
        # Two different type selectors in the same namespace can never
        # match the same element.
        return _BaseConflict()
    return None
