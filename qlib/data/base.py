"""Expression foundations: AST parsing, validation and evaluation.

Expressions use a restricted AST interpreter: no eval, attribute access,
imports or arbitrary calls.
"""

import ast
from functools import lru_cache
import operator
import re

import numpy as np
import pandas as pd

from .ops import OPERATORS

FIELD = re.compile(r"\$\$([A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)?)|\$([A-Za-z_][A-Za-z_0-9]*)")
BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
       ast.BitAnd: operator.and_, ast.BitOr: operator.or_}
CMP = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt,
       ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}


@lru_cache(maxsize=512)
def parse(expression):
    if not isinstance(expression, str) or len(expression) > 10000:
        raise ValueError("Expression must be a string of at most 10000 characters")
    source = FIELD.sub(lambda m: f"pitfield('{m[1]}')" if m[1] else f"field('{m[2]}')", expression)
    try:
        tree = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expression}") from exc
    if sum(1 for _ in ast.walk(tree)) > 1000:
        raise ValueError("Expression is too complex")
    return tree


def literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -literal(node.operand)
    raise ValueError("Offsets must be numeric literals")


def validate(expression, allow_future=True):
    tree = parse(expression)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ValueError("Only named operators with positional arguments are allowed")
            name = node.func.id
            if name not in {*OPERATORS, "field", "pitfield", "P", "PRef"}:
                raise ValueError(f"Unknown operator: {name}")
            if name in ("field", "pitfield"):
                if (len(node.args) != 1 or not isinstance(node.args[0], ast.Constant)
                        or not isinstance(node.args[0].value, str)
                        or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)?"
                                            if name == "pitfield" else r"[A-Za-z_][A-Za-z_0-9]*", node.args[0].value)):
                    raise ValueError("Invalid field name")
            if name in ("Ref", "Delta") and not allow_future:
                if len(node.args) != 2 or literal(node.args[1]) < 0:
                    raise ValueError("Future Ref/Delta is forbidden in filters and strategies")
            if name in ("P", "PRef"):
                if len(node.args) != (1 if name == "P" else 2):
                    raise ValueError(f"Invalid {name} arguments")
                if name == "PRef":
                    offset = literal(node.args[1])
                    if isinstance(offset, bool) or not isinstance(offset, int) or offset > 0:
                        raise ValueError("PRef offset must be an integer <= 0")
                calls = [child for child in ast.walk(node.args[0]) if isinstance(child, ast.Call)]
                if not any(isinstance(child.func, ast.Name) and child.func.id == "pitfield" for child in calls):
                    raise ValueError("P/PRef requires a $$financial field")
                for child in calls:
                    if isinstance(child.func, ast.Name):
                        if child.func.id in ("field", "P", "PRef"):
                            raise ValueError("Daily fields and nested P/PRef are forbidden inside P/PRef")
                        if child.func.id in ("Ref", "Delta") and (len(child.args) != 2 or literal(child.args[1]) < 0):
                            raise ValueError("Future report periods are forbidden inside P/PRef")
        elif not isinstance(node, (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
                                   ast.Compare, ast.BoolOp, ast.operator, ast.unaryop,
                                   ast.cmpop, ast.boolop)):
            raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")
    return tree


class ExpressionEngine:
    def __init__(self, provider, instrument, calendar, allow_future=True):
        self.provider, self.instrument, self.calendar = provider, instrument, calendar
        self.allow_future = allow_future
        self.cache = {}

    def evaluate(self, expression):
        tree = validate(expression, self.allow_future)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self._prepare_pit([tree])
            result = self._eval(tree)
        return pd.Series(result, index=self.calendar).replace([np.inf, -np.inf], np.nan)

    def prepare(self, expressions):
        """Share one stock load and one event scan across all PIT projections."""
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self._prepare_pit([validate(expression, self.allow_future) for expression in expressions])

    def _prepare_pit(self, trees):
        projections = {}

        def collect(node):
            if isinstance(node, ast.Call) and node.func.id in ("P", "PRef", "pitfield"):
                key = ast.dump(node)
                if key not in self.cache:
                    inner = node if node.func.id == "pitfield" else node.args[0]
                    offset = literal(node.args[1]) if node.func.id == "PRef" else 0
                    names = {child.args[0].value for child in ast.walk(inner)
                             if isinstance(child, ast.Call) and child.func.id == "pitfield"}
                    projections[key] = inner, offset, names
            else:
                for child in ast.iter_child_nodes(node):
                    collect(child)

        for tree in trees:
            collect(tree)
        if not projections:
            return
        store = self.provider._pit
        names = set().union(*(spec[2] for spec in projections.values()))
        ids = {name: store.resolve(name) for name in names}
        stock = store.stock(self.instrument)
        results = {key: [] for key in projections}
        records = stock.records[np.isin(stock.records["field_id"], list(ids.values()))]
        if len(self.calendar) and len(records):
            periods = records["period"].astype(np.int64)
            ordinal = periods // 100 * 4 + periods % 100 - 1
            first, last = int(ordinal.min()), int(ordinal.max())
            grid = np.arange(first, last + 1)
            labels = pd.Index(grid // 4 * 100 + grid % 4 + 1, name="period")
            current = {field_id: np.full(len(grid), np.nan) for field_id in set(ids.values())}
            known = {field_id: np.zeros(len(grid), dtype=bool) for field_id in current}
            for date, updates in stock.events(list(current), self.calendar[-1]):
                changed = set(map(int, updates["field_id"]))
                for row in updates:
                    field_id, period = int(row["field_id"]), int(row["period"])
                    slot = period // 100 * 4 + period % 100 - 1 - first
                    current[field_id][slot] = row["value"]
                    known[field_id][slot] = True
                for key, (inner, offset, fields) in projections.items():
                    if not changed.intersection(ids[name] for name in fields):
                        continue
                    visible = np.flatnonzero(np.logical_or.reduce([known[ids[name]] for name in fields]))
                    start, stop = int(visible[0]), int(visible[-1]) + 1
                    context = {name: pd.Series(current[ids[name]][start:stop], index=labels[start:stop]) for name in fields}
                    value = pd.Series(self._eval(inner, context), index=labels[start:stop])
                    position = len(value) - 1 + offset
                    results[key].append((pd.Timestamp(str(date)), value.iloc[position] if position >= 0 else np.nan))
        for key, events in results.items():
            if events:
                dates, values = zip(*events)
                # Reindex the event series itself: explicit NaN events invalidate
                # previous values, unlike a value-level ffill().
                self.cache[key] = pd.Series(values, index=pd.DatetimeIndex(dates)).reindex(self.calendar, method="ffill")
            else:
                self.cache[key] = pd.Series(np.nan, index=self.calendar)

    def _eval(self, node, pit=None):
        key = ast.dump(node)
        if pit is None and key in self.cache:
            return self.cache[key]
        result = self._visit(node, pit)
        if pit is None:
            self.cache[key] = result
        return result

    def _visit(self, node, pit=None):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in BIN:
            return BIN[type(node.op)](self._eval(node.left, pit), self._eval(node.right, pit))
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, pit)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, (ast.Invert, ast.Not)):
                return np.logical_not(value)
        if isinstance(node, ast.Compare):
            left, result = self._eval(node.left, pit), True
            for op, right_node in zip(node.ops, node.comparators):
                if type(op) not in CMP:
                    raise ValueError("Unsupported comparison")
                right = self._eval(right_node, pit)
                result = result & CMP[type(op)](left, right)
                left = right
            return result
        if isinstance(node, ast.BoolOp):
            values = [self._eval(x, pit) for x in node.values]
            result = values[0]
            for value in values[1:]:
                result = result & value if isinstance(node.op, ast.And) else result | value
            return result
        if isinstance(node, ast.Call):
            name = node.func.id
            if name == "field":
                return self.provider._field(self.instrument, node.args[0].value).reindex(self.calendar)
            if name == "pitfield":
                return pit[node.args[0].value]
            return OPERATORS[name](*(self._eval(arg, pit) for arg in node.args))
        raise ValueError(f"Unsupported expression syntax: {ast.dump(node)}")
