"""A small, safe expression language for formula actions and computed attributes.

Supports numbers, 'strings', identifiers, + - * / %, comparisons, && || !, ternary ?: and a
whitelist of functions. It is a hand-written tokenizer and recursive-descent parser: there is
no eval/exec, no Python AST evaluation, and no attribute access. Identifiers are only looked up
as keys in the supplied variables mapping.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any, Callable, Mapping, Union

Scalar = Union[bool, int, float, str, None]


class ExpressionError(ValueError):
    pass


_OPS = ("<=", ">=", "==", "!=", "&&", "||", "+", "-", "*", "/", "%", "<", ">", "!", "(", ")", ",", "?", ":")
_ID_START = re.compile(r"[A-Za-z_]")
_ID_CHAR = re.compile(r"[A-Za-z0-9_.]")
_NUM_CHAR = re.compile(r"[0-9.]")
MAX_LENGTH = 500
MAX_DEPTH = 40


@dataclass(frozen=True)
class Tok:
    t: str  # num | str | id | op
    v: Any


def _tokenize(src: str) -> list[Tok]:
    if len(src) > MAX_LENGTH:
        raise ExpressionError(f"Expression longer than {MAX_LENGTH} characters")
    out: list[Tok] = []
    i = 0
    while i < len(src):
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if _NUM_CHAR.match(c):
            j = i
            while j < len(src) and _NUM_CHAR.match(src[j]):
                j += 1
            try:
                out.append(Tok("num", float(src[i:j])))
            except ValueError:
                raise ExpressionError(f'Bad number "{src[i:j]}"') from None
            i = j
            continue
        if c in ("'", '"'):
            j = src.find(c, i + 1)
            if j < 0:
                raise ExpressionError("Unterminated string")
            out.append(Tok("str", src[i + 1 : j]))
            i = j + 1
            continue
        if _ID_START.match(c):
            j = i
            while j < len(src) and _ID_CHAR.match(src[j]):
                j += 1
            name = src[i:j]
            if "__" in name:
                raise ExpressionError(f'Identifier "{name}" is not allowed')
            out.append(Tok("id", name))
            i = j
            continue
        op = next((o for o in _OPS if src.startswith(o, i)), None)
        if op is None:
            raise ExpressionError(f'Unexpected character "{c}"')
        out.append(Tok("op", op))
        i += len(op)
    return out


# AST nodes are plain tuples: ("lit", v) ("var", name) ("un", op, a) ("bin", op, a, b)
# ("tern", c, a, b) ("call", fn, [args])


class _Parser:
    def __init__(self, toks: list[Tok]):
        self.toks = toks
        self.p = 0
        self.depth = 0

    def peek(self) -> Tok | None:
        return self.toks[self.p] if self.p < len(self.toks) else None

    def is_op(self, v: str) -> bool:
        t = self.peek()
        return t is not None and t.t == "op" and t.v == v

    def expect(self, v: str) -> None:
        if not self.is_op(v):
            raise ExpressionError(f'Expected "{v}"')
        self.p += 1

    def _binary(self, nxt: Callable[[], tuple], ops: tuple[str, ...]) -> tuple:
        left = nxt()
        while True:
            t = self.peek()
            if t is None or t.t != "op" or t.v not in ops:
                return left
            self.p += 1
            left = ("bin", t.v, left, nxt())

    def primary(self) -> tuple:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExpressionError("Expression is nested too deeply")
        try:
            tok = self.peek()
            if tok is None:
                raise ExpressionError("Unexpected end of expression")
            self.p += 1
            if tok.t in ("num", "str"):
                return ("lit", tok.v)
            if tok.t == "id":
                if tok.v in ("true", "false"):
                    return ("lit", tok.v == "true")
                if self.is_op("("):
                    self.p += 1
                    args: list[tuple] = []
                    if not self.is_op(")"):
                        args.append(self.ternary())
                        while self.is_op(","):
                            self.p += 1
                            args.append(self.ternary())
                    self.expect(")")
                    return ("call", tok.v, args)
                return ("var", tok.v)
            if tok.v == "(":
                e = self.ternary()
                self.expect(")")
                return e
            if tok.v in ("-", "!", "+"):
                return ("un", tok.v, self.primary())
            raise ExpressionError(f'Unexpected "{tok.v}"')
        finally:
            self.depth -= 1

    def mul(self) -> tuple:
        return self._binary(self.primary, ("*", "/", "%"))

    def add(self) -> tuple:
        return self._binary(self.mul, ("+", "-"))

    def cmp(self) -> tuple:
        return self._binary(self.add, ("<", "<=", ">", ">="))

    def eq(self) -> tuple:
        return self._binary(self.cmp, ("==", "!="))

    def and_(self) -> tuple:
        return self._binary(self.eq, ("&&",))

    def or_(self) -> tuple:
        return self._binary(self.and_, ("||",))

    def ternary(self) -> tuple:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExpressionError("Expression is nested too deeply")
        try:
            c = self.or_()
            if self.is_op("?"):
                self.p += 1
                a = self.ternary()
                self.expect(":")
                return ("tern", c, a, self.ternary())
            return c
        finally:
            self.depth -= 1


@lru_cache(maxsize=1024)
def _compile(src: str) -> tuple:
    parser = _Parser(_tokenize(src))
    root = parser.ternary()
    if parser.p < len(parser.toks):
        raise ExpressionError(f'Unexpected "{parser.toks[parser.p].v}"')
    return root


# ---------- runtime ----------


def parse_iso_date(v: Scalar) -> date:
    if not isinstance(v, str) or not re.match(r"^\d{4}-\d{2}-\d{2}", v):
        raise ExpressionError(f"Not a date: {v!r}")
    try:
        return date.fromisoformat(v[:10])
    except ValueError:
        raise ExpressionError(f"Not a date: {v!r}") from None


def num(v: Scalar) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ExpressionError(f"Not a number: {v!r}") from None


def js_str(v: Any) -> str:
    """String form matching the reference (JavaScript) implementation."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _loose_eq(a: Scalar, b: Scalar) -> bool:
    if type(a) is type(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool)):
        return a == b
    return js_str(a) == js_str(b)


def _round(a: Scalar, d: Scalar = 0) -> float:
    f = 10 ** num(d)
    return math.floor(num(a) * f + 0.5) / f


def _weekday_sun0(a: Scalar) -> int:
    return (parse_iso_date(a).weekday() + 1) % 7


_FUNCS: dict[str, Callable[..., Scalar]] = {
    "min": lambda *a: min(num(x) for x in a),
    "max": lambda *a: max(num(x) for x in a),
    "abs": lambda a: abs(num(a)),
    "floor": lambda a: float(math.floor(num(a))),
    "ceil": lambda a: float(math.ceil(num(a))),
    "round": _round,
    "clamp": lambda x, lo, hi: min(max(num(x), num(lo)), num(hi)),
    "if": lambda c, a, b: a if c else b,
    "daysBetween": lambda a, b: float((parse_iso_date(b) - parse_iso_date(a)).days),
    "dayOfWeek": lambda a: float(_weekday_sun0(a)),  # 0 = Sunday … 6 = Saturday
    "month": lambda a: float(parse_iso_date(a).month),
    "isWeekend": lambda a: _weekday_sun0(a) in (0, 6),
}
EXPRESSION_FUNCTIONS = tuple(_FUNCS)


def _eval(n: tuple, vars_: Mapping[str, Any]) -> Scalar:
    k = n[0]
    if k == "lit":
        return n[1]
    if k == "var":
        if n[1] not in vars_:
            raise ExpressionError(f'Unknown variable "{n[1]}"')
        v = vars_[n[1]]
        if isinstance(v, (list, tuple)):
            return ",".join(js_str(x) for x in v)
        if not isinstance(v, (bool, int, float, str)) and v is not None:
            raise ExpressionError(f'Variable "{n[1]}" is not a scalar')
        return v
    if k == "un":
        a = _eval(n[2], vars_)
        if n[1] == "!":
            return not a
        return -num(a) if n[1] == "-" else num(a)
    if k == "tern":
        return _eval(n[2], vars_) if _eval(n[1], vars_) else _eval(n[3], vars_)
    if k == "call":
        fn = _FUNCS.get(n[1])
        if fn is None:
            raise ExpressionError(f'Unknown function "{n[1]}"')
        try:
            return fn(*[_eval(a, vars_) for a in n[2]])
        except TypeError:
            raise ExpressionError(f'Wrong number of arguments for "{n[1]}"') from None
    if k == "bin":
        op = n[1]
        if op == "&&":
            return bool(_eval(n[2], vars_)) and bool(_eval(n[3], vars_))
        if op == "||":
            return bool(_eval(n[2], vars_)) or bool(_eval(n[3], vars_))
        a, b = _eval(n[2], vars_), _eval(n[3], vars_)
        if op == "==":
            return _loose_eq(a, b)
        if op == "!=":
            return not _loose_eq(a, b)
        if op == "<":
            return num(a) < num(b)
        if op == "<=":
            return num(a) <= num(b)
        if op == ">":
            return num(a) > num(b)
        if op == ">=":
            return num(a) >= num(b)
        if op == "+":
            return num(a) + num(b)
        if op == "-":
            return num(a) - num(b)
        if op == "*":
            return num(a) * num(b)
        if op in ("/", "%"):
            if num(b) == 0:
                raise ExpressionError("Division by zero")
            return num(a) / num(b) if op == "/" else math.fmod(num(a), num(b))
    raise ExpressionError("Invalid expression")


def evaluate_expression(src: str, vars_: Mapping[str, Any]) -> Scalar:
    return _eval(_compile(src), vars_)


def expression_variables(src: str) -> list[str]:
    out: list[str] = []

    def walk(n: tuple) -> None:
        k = n[0]
        if k == "var" and n[1] not in out:
            out.append(n[1])
        elif k == "un":
            walk(n[2])
        elif k == "bin":
            walk(n[2])
            walk(n[3])
        elif k == "tern":
            walk(n[1])
            walk(n[2])
            walk(n[3])
        elif k == "call":
            for a in n[2]:
                walk(a)

    walk(_compile(src))
    return out


def expression_functions(src: str) -> list[str]:
    out: list[str] = []

    def walk(n: tuple) -> None:
        k = n[0]
        if k == "call":
            out.append(n[1])
            for a in n[2]:
                walk(a)
        elif k == "un":
            walk(n[2])
        elif k == "bin":
            walk(n[2])
            walk(n[3])
        elif k == "tern":
            walk(n[1])
            walk(n[2])
            walk(n[3])

    walk(_compile(src))
    return out


def check_expression(src: str) -> str | None:
    """Return None when the expression parses and only calls known functions, else an error message."""
    try:
        unknown = [f for f in expression_functions(src) if f not in _FUNCS]
    except ExpressionError as e:
        return str(e)
    if unknown:
        return f'Unknown function "{unknown[0]}"'
    return None
