"""
cssselect.optimize
==================

Optional simplification of parsed CSS selector trees.

The optimizer only applies rewrites whose equivalence is provable from the
structure of the tree: duplicate conditions, provably always-true or
always-false selector lists, and absorption between the compound selector a
functional pseudo-class applies to and its arguments. Unknown extensions,
different namespace bindings, :has() relations and functional pseudo-classes
are never guessed with.

The original parse tree is never modified: every rewrite builds new parsed
objects. The specificity reported by the returned selector is always that
of the input selector, even when the simplification removes conditions
that would otherwise contribute to it.

An expansion budget (``max_expansions``) bounds how many list members the
rewrites may emit, so that simplification cannot build exponentially large
intermediate trees.

:copyright: (c) 2007-2012 Ian Bicking and contributors.
See AUTHORS for more details.
:license: BSD, see LICENSE for more details.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    Tree,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The default cap on list members produced while rewriting one selector.
DEFAULT_MAX_EXPANSIONS = 32

# Rewrite kinds exposed in :class:`OptimizationRecord`.
DUP_CONDITION = "duplicate-condition"
ALWAYS_TRUE = "always-true"
ALWAYS_FALSE = "always-false"
ABSORPTION = "absorption"
#: Raised when the expansion budget is exhausted.
BUDGET_EXCEEDED = "budget-exceeded"


@dataclass(frozen=True)
class OptimizationRecord:
    """One simplification applied (or attempted) by :func:`optimize_selector`.

    .. attribute:: kind

        One of ``"duplicate-condition"``, ``"always-true"``,
        ``"always-false"``, ``"absorption"`` or ``"budget-exceeded"``.

    .. attribute:: detail

        A human-readable description of the rewrite, naming the canonical
        form of the simplified construct.

    """

    kind: str
    detail: str


@dataclass(frozen=True)
class OptimizationResult:
    """The outcome of :func:`optimize_selector`.

    .. attribute:: selector

        A new :class:`~cssselect.Selector` object. The parse tree of the
        input selector is never modified. Its specificity is identical to
        that of the input selector.

    .. attribute:: records

        The simplifications that were applied, in application order.

    """

    selector: Selector
    records: tuple[OptimizationRecord, ...]


def optimize_selector(
    selector: Selector, *, max_expansions: int = DEFAULT_MAX_EXPANSIONS
) -> OptimizationResult:
    """Return a simplified copy of a parsed selector.

    Only provably equivalent rewrites are applied: exact duplicate
    conditions, selector lists that are provably always true or always
    false, and absorption between a compound selector and the arguments of
    ``:is()``/``:where()``/``:not()``/``:has()`` applying to it. Anything
    the optimizer cannot prove equivalent — unknown extensions, different
    namespace bindings, functional pseudo-classes, ``:has()`` relations
    with combinators — is left untouched.

    :param selector:
        A parsed :class:`~cssselect.Selector` object.
    :param max_expansions:
        Maximum number of list members the rewrites may emit. Once the
        budget is exhausted the remaining tree is returned as is and a
        ``"budget-exceeded"`` record is appended.
    :returns:
        An :class:`OptimizationResult` with the optimized selector and the
        records of the simplifications that were applied.

    """
    if max_expansions < 1:
        raise ValueError(f"max_expansions must be >= 1, got {max_expansions}")
    optimizer = _Optimizer(max_expansions)
    tree = optimizer.optimize(selector.parsed_tree)
    optimized = _OptimizedSelector(
        tree, selector.pseudo_element, selector.specificity()
    )
    return OptimizationResult(optimized, tuple(optimizer.records))


class _OptimizedSelector(Selector):
    """A selector whose specificity is that of the selector it was
    optimized from.

    Simplification removes redundant conditions, which per the CSS
    specification still contribute to specificity, so the specificity of
    the original selector is kept instead of being recomputed from the
    simplified tree.
    """

    def __init__(
        self,
        tree: Tree,
        pseudo_element: PseudoElement | None,
        specificity: tuple[int, int, int],
    ) -> None:
        super().__init__(tree, pseudo_element)
        self._specificity = specificity

    def specificity(self) -> tuple[int, int, int]:
        return self._specificity


#### Structural keys


def _strip_universal_star(canonical: str) -> str:
    # Like Selector.canonical() and friends: "*.foo" and ".foo" are the
    # same condition.
    if len(canonical) > 1 and canonical[0] == "*" and canonical[1] in "#.[:":
        return canonical[1:]
    return canonical


def _key(tree: Tree) -> str | None:
    """A canonical string identifying the condition a tree tests, or None.

    The key of a compound selector is its normalized canonical CSS form
    (the parser guarantees that canonical output round-trips to the same
    tree), so two trees with the same key always test the same condition
    on the same element. Trees whose translation depends on state outside
    the tree — namespace bindings, :scope, functional pseudo-classes or
    combinators — return None and are never rewritten.
    """
    if isinstance(tree, CombinedSelector):
        # Combinators are position-dependent, so never compared.
        return None
    if _uses_unknown_state(tree):
        return None
    return _strip_universal_star(tree.canonical())


def _uses_unknown_state(tree: Tree) -> bool:
    """True if translating *tree* depends on something the optimizer cannot
    prove from the tree itself (namespace bindings, :scope or a functional
    pseudo-class)."""
    if isinstance(tree, Element):
        return tree.namespace is not None
    if isinstance(tree, Attrib):
        return tree.namespace is not None or _uses_unknown_state(tree.selector)
    if isinstance(tree, (Hash, Class, Pseudo)):
        if isinstance(tree, Pseudo) and tree.ident == "scope":
            return True
        return _uses_unknown_state(tree.selector)
    if isinstance(tree, Function):
        # Raw-token arguments and positional behavior are not analyzed.
        return True
    if isinstance(tree, Negation):
        return (
            _uses_unknown_state(tree.selector)
            or _uses_unknown_state(tree.subselector)
        )
    if isinstance(tree, (Matching, SpecificityAdjustment)):
        return _uses_unknown_state(tree.selector) or any(
            _uses_unknown_state(subselector) for subselector in tree.selector_list
        )
    if isinstance(tree, Relation):
        return _uses_unknown_state(tree.selector) or any(
            _uses_unknown_state(subselector.parsed_tree)
            for _, subselector in tree.arguments
        )
    return False  # pragma: no cover - all Tree cases handled above


#### Compound selectors


def _flatten(tree: Tree) -> tuple[Element, list[Tree]] | None:
    """Split a compound selector into its base Element and conditions.

    Returns None if the tree is not a compound selector (i.e. contains a
    combinator).
    """
    conditions: list[Tree] = []
    while not isinstance(tree, Element):
        if isinstance(tree, (Hash, Class, Attrib, Pseudo, Negation, Function)):
            conditions.append(tree)
            tree = tree.selector
        elif isinstance(tree, (Matching, SpecificityAdjustment)):
            conditions.append(tree)
            tree = tree.selector
        elif isinstance(tree, Relation):
            conditions.append(tree)
            tree = tree.selector
        else:
            return None
    conditions.reverse()
    return tree, conditions


def _rebuild(base: Element, conditions: Sequence[Tree]) -> Tree:
    """Reassemble a compound selector from its base Element and conditions.

    The new nodes wrap the *base* element itself, so the specificity of the
    result is the sum of the specificities of the parts.
    """
    tree: Tree = base
    for condition in conditions:
        if isinstance(condition, Hash):
            tree = Hash(tree, condition.id)
        elif isinstance(condition, Class):
            tree = Class(tree, condition.class_name)
        elif isinstance(condition, Attrib):
            tree = Attrib(
                tree,
                condition.namespace,
                condition.attrib,
                condition.operator,
                condition.value,
                condition.flag,
            )
        elif isinstance(condition, Pseudo):
            tree = Pseudo(tree, condition.ident)
        elif isinstance(condition, Negation):
            tree = Negation(tree, condition.subselector)
        elif isinstance(condition, Function):
            tree = Function(tree, condition.name, condition.arguments)
        elif isinstance(condition, Matching):
            tree = Matching(tree, condition.selector_list)
        elif isinstance(condition, SpecificityAdjustment):
            tree = SpecificityAdjustment(tree, condition.selector_list)
        elif isinstance(condition, Relation):
            tree = Relation(tree, condition.arguments)
        else:  # pragma: no cover - _flatten only yields the types above
            raise AssertionError(f"Cannot rebuild {condition!r}")
    return tree


def _describe(tree: Tree) -> str:
    return tree.canonical()


class _Optimizer:
    def __init__(self, max_expansions: int) -> None:
        self.max_expansions = max_expansions
        self.expansions = 0
        self.budget_exceeded = False
        self.records: list[OptimizationRecord] = []

    def _record(self, kind: str, detail: str) -> None:
        self.records.append(OptimizationRecord(kind, detail))

    def _charge(self, count: int) -> bool:
        """Account for *count* emitted list members.

        Returns False if the budget is exceeded: the caller must abort the
        rewrite and keep the tree it was asked to simplify.
        """
        self.expansions += count
        if self.expansions > self.max_expansions:
            self.budget_exceeded = True
            return False
        return True

    def _optimize_condition(self, condition: Tree) -> Tree:
        if isinstance(condition, Negation):
            subselector = self.optimize(condition.subselector)
            return Negation(condition.selector, subselector)
        if isinstance(condition, Matching):
            selector_list = [
                self.optimize(subselector) for subselector in condition.selector_list
            ]
            return Matching(
                condition.selector, self._dedupe_list(selector_list, ":is()")
            )
        if isinstance(condition, SpecificityAdjustment):
            selector_list = [
                self.optimize(subselector) for subselector in condition.selector_list
            ]
            return SpecificityAdjustment(
                condition.selector, self._dedupe_list(selector_list, ":where()")
            )
        if isinstance(condition, Relation):
            arguments = [
                (combinator, Selector(self.optimize(subselector.parsed_tree)))
                for combinator, subselector in condition.arguments
            ]
            return Relation(condition.selector, arguments)
        return condition

    def _dedupe_list(self, selector_list: list[Tree], name: str) -> list[Tree]:
        """Drop exact duplicate members of an :is()/:where() list."""
        seen: set[str] = set()
        unique: list[Tree] = []
        for subselector in selector_list:
            key = _key(subselector)
            if key is None or key not in seen:
                seen.add(key or f"#{id(subselector)}")
                unique.append(subselector)
        dropped = len(selector_list) - len(unique)
        if dropped:
            self._record(
                DUP_CONDITION,
                f"removed {dropped} duplicate {name} argument(s) "
                f"in {_describe_list(selector_list)}",
            )
        return unique

    def optimize(self, tree: Tree) -> Tree:
        if isinstance(tree, CombinedSelector):
            return CombinedSelector(
                self.optimize(tree.selector),
                tree.combinator,
                self.optimize(tree.subselector),
            )
        flattened = _flatten(tree)
        if flattened is None:  # pragma: no cover - non-combined Trees are compounds
            return tree
        base, conditions = flattened
        if not conditions:
            return tree
        base, conditions = self._optimize_compound(base, conditions)
        return _rebuild(base, conditions)

    def _optimize_compound(
        self, base: Element, conditions: list[Tree]
    ) -> tuple[Element, list[Tree]]:
        optimized = [
            self._optimize_condition(condition) for condition in conditions
        ]
        # Compound-level rewrites run on the optimized children, iteratively
        # until nothing else applies (or the expansion budget is exhausted).
        while not self.budget_exceeded:
            records_before = len(self.records)
            base, optimized = self._simplify_compound(base, optimized)
            if len(self.records) == records_before:
                break
        return base, optimized

    def _simplify_compound(
        self, base: Element, conditions: list[Tree]
    ) -> tuple[Element, list[Tree]]:
        """Run one pass of the compound-level rewrites."""
        base, conditions = self._remove_duplicate_conditions(base, conditions)
        base, conditions = self._simplify_negations(base, conditions)
        base, conditions = self._simplify_matching_lists(base, conditions)
        base, conditions = self._simplify_relations(base, conditions)
        return base, conditions

    def _condition_key(
        self, base: Element, prefix: str, condition: Tree
    ) -> str | None:
        """Canonical key of the suffix *condition* applied after *prefix*.

        A compound flattened into (base, conditions) is a chain of nested
        wrappers; each condition's key is the canonical CSS it adds to the
        prefix built so far (after stripping a redundant universal
        selector), so e.g. both classes of ``div.a.a`` key to ``.a``.
        """
        full = _key(condition)
        if full is None:
            return None
        stripped_prefix = _strip_universal_star(prefix)
        if stripped_prefix == "*":
            # A bare universal selector contributes no condition.
            stripped_prefix = ""
        if full == stripped_prefix:
            # The condition contributes nothing beyond the prefix, e.g.
            # a bare universal selector.
            return ""
        if not full.startswith(stripped_prefix):
            # Keying mismatch (should not happen for flattened compounds).
            return None
        suffix = full[len(stripped_prefix) :]
        return _strip_universal_star(suffix) if suffix.startswith("*") else suffix

    def _condition_keys(
        self, base: Element, conditions: Sequence[Tree]
    ) -> list[str | None]:
        """Suffix keys of all conditions of a flattened compound."""
        prefix = base.canonical()
        keys: list[str | None] = []
        for condition in conditions:
            key = self._condition_key(base, prefix, condition)
            keys.append(key)
            if key is not None:
                prefix = _key(condition) or prefix
        return keys

    def _remove_duplicate_conditions(
        self, base: Element, conditions: list[Tree]
    ) -> tuple[Element, list[Tree]]:
        """Drop conditions that test exactly the same thing twice."""
        keys = self._condition_keys(base, conditions)
        seen: set[str] = set()
        unique: list[Tree] = []
        dropped: list[str] = []
        for condition, key in zip(conditions, keys, strict=True):
            if key is None or key not in seen:
                seen.add(key or f"#{id(condition)}")
                unique.append(condition)
            else:
                dropped.append(key or _describe(condition))
        if dropped:
            self._record(
                DUP_CONDITION,
                f"removed duplicate condition(s) {', '.join(dropped)}",
            )
        return base, unique

    def _simplify_negations(
        self, base: Element, conditions: list[Tree]
    ) -> tuple[Element, list[Tree]]:
        """Apply the provable :not() equivalences.

        * ``x:not(y)`` where the argument only tests conditions the
          compound already tests is redundant: the negation is removed
          (absorption).
        * ``x:not(y)`` where the compound satisfies every condition of the
          argument never matches (always-false), and is rewritten to the
          canonical ``:not(*)`` form.
        """
        result: list[Tree] = []
        for condition in conditions:
            if not isinstance(condition, Negation):
                result.append(condition)
                continue
            sub = condition.subselector
            if isinstance(sub, Element):
                # A bare element argument (including the universal ":not(*)")
                # has no compound conditions to apply the rules to.
                result.append(condition)
                continue
            sub_flattened = _flatten(sub)
            if sub_flattened is None:
                result.append(condition)
                continue
            sub_base, sub_conditions = sub_flattened
            # The rules compare the argument with the *rest* of the compound.
            others = [c for c in conditions if c is not condition]
            outer_keys = self._suffix_set(base, others)
            inner_keys = self._suffix_set(sub_base, sub_conditions)
            if outer_keys is None or inner_keys is None:
                result.append(condition)
                continue
            always_false = self._element_compatible(
                sub_base, base
            ) and inner_keys.issubset(outer_keys)
            if always_false:
                # Every element of the compound matches the argument (this
                # includes the equal case), so the negation excludes them
                # all.
                self._record(
                    ALWAYS_FALSE,
                    f":not({_describe(sub)}) covers the selector, never matches",
                )
                result.append(Negation(condition.selector, Element()))
                continue
            is_strict_subset = self._element_compatible(
                sub_base, base
            ) and inner_keys.issubset(outer_keys) and not self._same_compound(
                base, outer_keys, sub_base, inner_keys
            )
            if is_strict_subset:
                # The argument is a strict subset of the compound, so it
                # only excludes elements the compound already excludes:
                # the negation is redundant and can be removed. Equal sets
                # are the always-false case handled above.
                self._record(
                    ABSORPTION,
                    f":not({_describe(sub)}) is implied by the selector, removed",
                )
                continue
            result.append(condition)
        return base, result

    def _simplify_matching_lists(
        self, base: Element, conditions: list[Tree]
    ) -> tuple[Element, list[Tree]]:
        """Apply the provable :is()/:where() equivalences.

        * a universal member makes the list always-true: drop the condition.
        * a member equal to the rest of the compound makes the list
          always-true: drop the condition.
        * a single member absorbing the rest of the compound and covering
          every other member replaces the whole list (absorption).
        """
        result: list[Tree] = []
        for condition in conditions:
            if not isinstance(condition, (Matching, SpecificityAdjustment)):
                result.append(condition)
                continue
            name = "is" if isinstance(condition, Matching) else "where"
            others = [c for c in conditions if c is not condition]
            members = list(condition.selector_list)
            if any(self._is_universal(member) for member in members):
                self._record(
                    ALWAYS_TRUE,
                    f":{name}() with a universal argument always matches, removed",
                )
                continue
            if any(
                self._compound_equals(base, others, member) for member in members
            ):
                self._record(
                    ALWAYS_TRUE,
                    f":{name}() argument equal to the selector, removed",
                )
                continue
            absorbing = [
                member
                for member in members
                if self._compound_absorbs(base, others, member)
            ]
            if (
                len(absorbing) == 1
                and self._tree_absorbs(
                    absorbing[0], _rebuild(base, others)
                )
                and all(
                    self._tree_absorbs(absorbing[0], member)
                    for member in members
                )
            ):
                # The single absorbing member matches everything the list
                # matched: it replaces the whole list. It is kept as is, so
                # its specificity is preserved exactly.
                if not self._charge(len(members)):
                    self._record(
                        BUDGET_EXCEEDED,
                        f"expansion budget exceeded simplifying :{name}()",
                    )
                    result.append(condition)
                    continue
                self._record(
                    ABSORPTION,
                    f":{name}() argument {_describe(absorbing[0])} absorbs "
                    f"the others, list reduced to it",
                )
                result.append(absorbing[0])
                continue
            result.append(condition)
        return base, result

    def _tree_absorbs(self, outer: Tree, inner: Tree) -> bool:
        """True if every element matching *inner* also matches *outer*."""
        outer_flat = _flatten(outer)
        inner_flat = _flatten(inner)
        if outer_flat is None or inner_flat is None:
            return False
        if not self._element_compatible(outer_flat[0], inner_flat[0]):
            return False
        outer_keys = self._suffix_set(outer_flat[0], outer_flat[1])
        inner_keys = self._suffix_set(inner_flat[0], inner_flat[1])
        if outer_keys is None or inner_keys is None:
            return False
        return inner_keys.issubset(outer_keys)

    def _simplify_relations(
        self, base: Element, conditions: list[Tree]
    ) -> tuple[Element, list[Tree]]:
        """Drop duplicate arguments of :has() lists.

        Only plain (implicit descendant) arguments are considered: relative
        selectors with explicit combinators are complex relations the
        optimizer does not guess with.
        """
        result: list[Tree] = []
        for condition in conditions:
            if not isinstance(condition, Relation):
                result.append(condition)
                continue
            seen: set[str] = set()
            arguments = []
            dropped = 0
            for combinator, subselector in condition.arguments:
                key = None
                if combinator.value == " ":
                    key = _key(subselector.parsed_tree)
                if key is None or key not in seen:
                    seen.add(key or f"#{id(subselector)}")
                    arguments.append((combinator, subselector))
                else:
                    dropped += 1
            if dropped:
                self._record(
                    DUP_CONDITION,
                    f"removed {dropped} duplicate :has() argument(s)",
                )
            result.append(Relation(condition.selector, arguments))
        return base, result

    #### Predicates used by the rewrites above

    @staticmethod
    def _is_universal(tree: Tree) -> bool:
        return (
            isinstance(tree, Element)
            and tree.element is None
            and tree.namespace is None
        )

    def _suffix_set(
        self, base: Element, conditions: Sequence[Tree]
    ) -> set[str] | None:
        """The set of suffix keys of a flattened compound, or None if any
        condition is unkeyable (functional pseudo-class, :scope, namespace
        binding...), which makes the compound's condition set unknown."""
        keys = self._condition_keys(base, conditions)
        if any(key is None for key in keys):
            return None
        return {key for key in keys if key}

    def _element_compatible(self, outer: Element, inner: Element) -> bool:
        """True if the element test of *inner* matches everything the test
        of *outer* matches (i.e. the inner compound can absorb the outer).

        A universal outer test is absorbed by anything; a universal inner
        test does not absorb a typed outer one, since elements of other
        types match the inner test but not the outer one.
        """
        if outer.namespace is not None or inner.namespace is not None:
            # Different namespace bindings are never guessed with.
            return False
        if outer.element is None:
            return True
        if inner.element is None:
            return False
        return outer.element == inner.element

    @staticmethod
    def _same_compound(
        base: Element,
        outer_keys: set[str],
        sub_base: Element,
        inner_keys: set[str],
    ) -> bool:
        """True if two compounds provably match the same elements."""
        return (
            outer_keys == inner_keys
            and base.element == sub_base.element
            and base.namespace == sub_base.namespace
        )

    def _absorbs(
        self,
        outer_base: Element,
        outer_conditions: Sequence[Tree],
        inner_base: Element,
        inner_conditions: Sequence[Tree],
    ) -> bool:
        """True if the "inner" compound is provably a subset of "outer"."""
        if not self._element_compatible(outer_base, inner_base):
            return False
        outer_keys = self._suffix_set(outer_base, outer_conditions)
        inner_keys = self._suffix_set(inner_base, inner_conditions)
        if outer_keys is None or inner_keys is None:
            return False
        # Every condition the outer compound tests must be tested by the
        # inner one too.
        return outer_keys.issubset(inner_keys)

    def _compound_equals(
        self, base: Element, conditions: Sequence[Tree], tree: Tree
    ) -> bool:
        """True if *tree* provably matches the same elements as the
        compound made of *base* and *conditions*."""
        flattened = _flatten(tree)
        if flattened is None:
            return False
        tree_base, tree_conditions = flattened
        outer_keys = self._suffix_set(base, conditions)
        inner_keys = self._suffix_set(tree_base, tree_conditions)
        if outer_keys is None or inner_keys is None:
            return False
        return self._same_compound(base, outer_keys, tree_base, inner_keys)

    def _compound_absorbs(
        self, base: Element, conditions: Sequence[Tree], tree: Tree
    ) -> bool:
        flattened = _flatten(tree)
        if flattened is None:
            return False
        tree_base, tree_conditions = flattened
        return self._absorbs(base, conditions, tree_base, tree_conditions)


def _describe_list(trees: Sequence[Tree]) -> str:
    return ", ".join(_describe(tree) for tree in trees)
