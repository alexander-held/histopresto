"""Expressions for purefunc modifiers (pyhf PR 2717): a parser for the sympy-style syntax and a compiler
that turns all bindings of a workspace into one tape of elementary operations, evaluated by the C kernel.
Every tape entry ("slot") is a constant, a parameter or an operation on earlier slots.
"""

import math
import re

OP = {name: i for i, name in enumerate("const par add sub mul div pow neg sqrt exp log sin cos tan tanh abs asin acos atan sinh cosh".split())}
FUNCTIONS = {"sqrt": OP["sqrt"], "exp": OP["exp"], "log": OP["log"], "ln": OP["log"], "sin": OP["sin"], "cos": OP["cos"], "tan": OP["tan"],
             "tanh": OP["tanh"], "Abs": OP["abs"], "asin": OP["asin"], "acos": OP["acos"], "atan": OP["atan"], "sinh": OP["sinh"],
             "cosh": OP["cosh"]}  # fmt: skip
CONSTANTS = {"pi": math.pi, "E": math.e}
TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\*\*|[-+*/(),]))")


def parse(text):
    """syntax tree: ("num", value), ("name", name), ("tuple", items) or (opcode, argument[, argument])"""

    def fail(what):
        raise ValueError(f'The expression "{text}" cannot be read: {what}.')

    tokens = []
    at = 0
    end = len(text.rstrip())
    while at < end:
        m = TOKEN.match(text, at)
        if not m:
            fail(f"unexpected character at position {at + 1}")
        at = m.end()
        tokens.append(("num", float(m[1])) if m[1] is not None else ("name", m[2]) if m[2] is not None else ("op", m[3]))
    pos = 0

    def accept(v):
        nonlocal pos
        if pos < len(tokens) and tokens[pos] == ("op", v):
            pos += 1
            return True
        return False

    def expect(v):
        if not accept(v):
            fail(f'"{v}" expected')

    def list_():
        items = [sum_()]
        while accept(","):
            items.append(sum_())
        return items[0] if len(items) == 1 else ("tuple", items)

    def sum_():
        left = product()
        while True:
            if accept("+"):
                left = (OP["add"], left, product())
            elif accept("-"):
                left = (OP["sub"], left, product())
            else:
                return left

    def product():
        left = unary()
        while True:
            if accept("*"):
                left = (OP["mul"], left, unary())
            elif accept("/"):
                left = (OP["div"], left, unary())
            else:
                return left

    def unary():
        if accept("+"):
            return unary()
        if accept("-"):
            arg = unary()
            return ("num", -arg[1]) if arg[0] == "num" else (OP["neg"], arg)
        base = atom()
        return (OP["pow"], base, unary()) if accept("**") else base

    def atom():
        nonlocal pos
        if pos >= len(tokens):
            fail("it ends too early")
        kind, v = tokens[pos]
        pos += 1
        if kind == "num":
            return ("num", v)
        if kind == "name":
            if not accept("("):
                return ("num", CONSTANTS[v]) if v in CONSTANTS else ("name", v)
            if v not in FUNCTIONS:
                fail(f'the function "{v}" is not supported (supported are {", ".join(FUNCTIONS)})')
            arg = sum_()
            expect(")")
            return (FUNCTIONS[v], arg)
        if v != "(":
            fail(f'unexpected "{v}"')
        inner = list_()
        expect(")")
        return inner

    tree = list_()
    if pos < len(tokens):
        fail(f'unexpected "{tokens[pos][1]}"')
    return tree


def _walk(node):
    yield node
    for child in node[1] if node[0] == "tuple" else node[1:] if isinstance(node[0], int) else ():
        yield from _walk(child)


def parse_bindings(bindings):
    """bindings as in the workspace: [{names, expression, language}]
    returns (trees: {binding name: syntax tree}, symbols: sorted names of the parameters)"""
    trees = {}
    for binding in bindings:
        names = binding["names"]
        label = "The binding of " + ", ".join(f'"{n}"' for n in names)
        if binding["language"] != "sympy":
            raise ValueError(f'{label} uses the language "{binding["language"]}"; only "sympy" is supported.')
        tree = parse(binding["expression"])
        items = tree[1] if tree[0] == "tuple" else [tree]
        if len(items) != len(names) or (tree[0] == "tuple") != (len(names) > 1):
            raise ValueError(
                f"{label} declares {len(names)} name{'' if len(names) == 1 else 's'}, "
                f"but its expression has {len(items)} component{'' if len(items) == 1 else 's'}."
            )
        for name, item in zip(names, items):
            if name in trees:
                raise ValueError(f'The binding "{name}" is declared more than once.')
            if any(node[0] == "tuple" for node in _walk(item)):
                raise ValueError(f"{label} contains a nested tuple, which is not supported.")
            trees[name] = item
    symbols = set()
    state = {}  # binding name -> 1 while being followed, 2 when done

    def follow(name):
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise ValueError(f'The binding "{name}" refers to itself through other bindings.')
        state[name] = 1
        for node in _walk(trees[name]):
            if node[0] != "name":
                continue
            if node[1] in trees:
                follow(node[1])
            else:
                symbols.add(node[1])
        state[name] = 2

    for name in trees:
        follow(name)
    return trees, sorted(symbols)


def compile_bindings(trees, parameter):
    """tape for the bindings; parameter(name) gives the index of a symbol in the parameter vector
    returns (op, a, b, value, slots: {binding name: slot}, pars: {binding name: parameter indices it depends on})"""
    op, a, b, value, deps = [], [], [], [], []

    def push(o, x, y, v, d):
        op.append(o)
        a.append(x)
        b.append(y)
        value.append(v)
        deps.append(d)
        return len(op) - 1

    constants = {}
    parameters = {}
    slots = {}

    def emit(node):
        if node[0] == "num":
            if node[1] not in constants:
                constants[node[1]] = push(OP["const"], 0, 0, node[1], [])
            return constants[node[1]]
        if node[0] == "name":
            if node[1] in trees:
                return binding(node[1])
            p = parameter(node[1])
            if p not in parameters:
                parameters[p] = push(OP["par"], p, 0, 0.0, [p])
            return parameters[p]
        x = emit(node[1])
        exponent = node[2][1] if node[0] == OP["pow"] and node[2][0] == "num" else 0
        if float(exponent).is_integer() and 2 <= exponent <= 8:
            # small integer powers as products
            r = x
            for _ in range(1, int(exponent)):
                r = push(OP["mul"], r, x, 0.0, deps[x])
            return r
        y = emit(node[2]) if len(node) > 2 else 0
        return push(node[0], x, y, 0.0, list(dict.fromkeys(deps[x] + deps[y])) if len(node) > 2 else deps[x])

    def binding(name):
        if name not in slots:
            slots[name] = emit(trees[name])
        return slots[name]

    for name in trees:
        binding(name)
    return op, a, b, value, slots, {name: deps[slots[name]] for name in trees}
