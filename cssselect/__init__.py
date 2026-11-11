"""
CSS Selectors based on XPath
============================

This module supports selecting XML/HTML elements based on CSS selectors.

:copyright: (c) 2007-2012 Ian Bicking and contributors.
See AUTHORS for more details.
:license: BSD, see LICENSE for more details.

"""

from cssselect.optimize import (
    OptimizationRecord,
    OptimizationResult,
    optimize_selector,
)
from cssselect.parser import (
    FunctionalPseudoElement,
    Selector,
    SelectorError,
    SelectorSyntaxError,
    parse,
)
from cssselect.xpath import ExpressionError, GenericTranslator, HTMLTranslator

__all__ = (
    "ExpressionError",
    "FunctionalPseudoElement",
    "GenericTranslator",
    "HTMLTranslator",
    "OptimizationRecord",
    "OptimizationResult",
    "Selector",
    "SelectorError",
    "SelectorSyntaxError",
    "optimize_selector",
    "parse",
)

VERSION = "1.6.0"
__version__ = VERSION
