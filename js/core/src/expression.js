// Expressions for purefunc modifiers (pyhf PR 2717): a parser for the sympy-style syntax and a compiler
// that turns all bindings of a workspace into one tape of elementary operations. Every tape entry
// ("slot") is a constant, a parameter or an operation on earlier slots.

const OP = { const: 0, par: 1, add: 2, sub: 3, mul: 4, div: 5, pow: 6, neg: 7, sqrt: 8, exp: 9, log: 10, sin: 11, cos: 12, tan: 13, tanh: 14, abs: 15, asin: 16, acos: 17, atan: 18, sinh: 19, cosh: 20 };
const FUNCTIONS = { sqrt: OP.sqrt, exp: OP.exp, log: OP.log, ln: OP.log, sin: OP.sin, cos: OP.cos, tan: OP.tan, tanh: OP.tanh, Abs: OP.abs, asin: OP.asin, acos: OP.acos, atan: OP.atan, sinh: OP.sinh, cosh: OP.cosh };
const CONSTANTS = { pi: Math.PI, E: Math.E };
const TOKEN = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\*\*|[-+*/(),]))/y;

// syntax tree: ["num", value], ["name", name], ["tuple", items] or [opcode, argument, argument]
function parse(text) {
  const fail = (what) => {
    throw new Error(`The expression "${text}" cannot be read: ${what}.`);
  };
  const tokens = [];
  const end = text.trimEnd().length;
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < end) {
    const at = TOKEN.lastIndex;
    const m = TOKEN.exec(text);
    if (!m) fail(`unexpected character at position ${at + 1}`);
    tokens.push(m[1] !== undefined ? ["num", parseFloat(m[1])] : m[2] !== undefined ? ["name", m[2]] : ["op", m[3]]);
  }
  let pos = 0;
  const peek = (v) => pos < tokens.length && tokens[pos][0] === "op" && tokens[pos][1] === v;
  const accept = (v) => peek(v) && ++pos > 0;
  const expect = (v) => accept(v) || fail(`"${v}" expected`);

  const list = () => {
    const items = [sum()];
    while (accept(",")) items.push(sum());
    return items.length === 1 ? items[0] : ["tuple", items];
  };
  const sum = () => {
    let left = product();
    for (;;) {
      if (accept("+")) left = [OP.add, left, product()];
      else if (accept("-")) left = [OP.sub, left, product()];
      else return left;
    }
  };
  const product = () => {
    let left = unary();
    for (;;) {
      if (accept("*")) left = [OP.mul, left, unary()];
      else if (accept("/")) left = [OP.div, left, unary()];
      else return left;
    }
  };
  const unary = () => {
    if (accept("+")) return unary();
    if (accept("-")) {
      const arg = unary();
      return arg[0] === "num" ? ["num", -arg[1]] : [OP.neg, arg];
    }
    const base = atom();
    return accept("**") ? [OP.pow, base, unary()] : base;
  };
  const atom = () => {
    if (pos >= tokens.length) fail("it ends too early");
    const [kind, v] = tokens[pos++];
    if (kind === "num") return ["num", v];
    if (kind === "name") {
      if (!accept("(")) return v in CONSTANTS ? ["num", CONSTANTS[v]] : ["name", v];
      if (!(v in FUNCTIONS)) fail(`the function "${v}" is not supported (supported are ${Object.keys(FUNCTIONS).join(", ")})`);
      const arg = sum();
      expect(")");
      return [FUNCTIONS[v], arg];
    }
    if (v !== "(") fail(`unexpected "${v}"`);
    const inner = list();
    expect(")");
    return inner;
  };

  const tree = list();
  if (pos < tokens.length) fail(`unexpected "${tokens[pos][1]}"`);
  return tree;
}

const walk = (node, visit) => {
  visit(node);
  if (node[0] === "tuple") node[1].forEach((n) => walk(n, visit));
  else if (typeof node[0] === "number") node.slice(1).forEach((n) => walk(n, visit));
};

// bindings as in the workspace: [{names, expression, language}]
// returns {trees: Map binding name -> syntax tree, symbols: sorted names of the parameters}
function parseBindings(bindings) {
  const trees = new Map();
  for (const binding of bindings) {
    const label = `The binding of ${binding.names.map((n) => `"${n}"`).join(", ")}`;
    if (binding.language !== "sympy") throw new Error(`${label} uses the language "${binding.language}"; only "sympy" is supported.`);
    const tree = parse(binding.expression);
    const items = tree[0] === "tuple" ? tree[1] : [tree];
    if (items.length !== binding.names.length || (tree[0] === "tuple") !== binding.names.length > 1) {
      throw new Error(`${label} declares ${binding.names.length} name${binding.names.length === 1 ? "" : "s"}, but its expression has ${items.length} component${items.length === 1 ? "" : "s"}.`);
    }
    binding.names.forEach((name, i) => {
      if (trees.has(name)) throw new Error(`The binding "${name}" is declared more than once.`);
      walk(items[i], (node) => {
        if (node[0] === "tuple") throw new Error(`${label} contains a nested tuple, which is not supported.`);
      });
      trees.set(name, items[i]);
    });
  }
  const symbols = new Set();
  const state = new Map(); // binding name -> 1 while being followed, 2 when done
  const follow = (name) => {
    if (state.get(name) === 2) return;
    if (state.get(name) === 1) throw new Error(`The binding "${name}" refers to itself through other bindings.`);
    state.set(name, 1);
    walk(trees.get(name), (node) => {
      if (node[0] !== "name") return;
      if (trees.has(node[1])) follow(node[1]);
      else symbols.add(node[1]);
    });
    state.set(name, 2);
  };
  for (const name of trees.keys()) follow(name);
  return { trees, symbols: [...symbols].sort() };
}

// tape for the bindings; parameter(name) gives the index of a symbol in the parameter vector
// returns {n, op, a, b, value, slots: binding name -> slot, pars: binding name -> parameter indices it depends on}
function compileBindings(trees, parameter) {
  const op = [];
  const a = [];
  const b = [];
  const value = [];
  const deps = [];
  const push = (o, x, y, v, d) => {
    op.push(o);
    a.push(x);
    b.push(y);
    value.push(v);
    deps.push(d);
    return op.length - 1;
  };
  const constants = new Map();
  const parameters = new Map();
  const slots = {};
  const emit = (node) => {
    if (node[0] === "num") {
      if (!constants.has(node[1])) constants.set(node[1], push(OP.const, 0, 0, node[1], []));
      return constants.get(node[1]);
    }
    if (node[0] === "name") {
      if (trees.has(node[1])) return binding(node[1]);
      const p = parameter(node[1]);
      if (!parameters.has(p)) parameters.set(p, push(OP.par, p, 0, 0, [p]));
      return parameters.get(p);
    }
    const x = emit(node[1]);
    const exponent = node[0] === OP.pow && node[2][0] === "num" ? node[2][1] : 0;
    if (Number.isInteger(exponent) && exponent >= 2 && exponent <= 8) {
      // small integer powers as products
      let r = x;
      for (let k = 1; k < exponent; k++) r = push(OP.mul, r, x, 0, deps[x]);
      return r;
    }
    const y = node.length > 2 ? emit(node[2]) : 0;
    return push(node[0], x, y, 0, node.length > 2 ? [...new Set([...deps[x], ...deps[y]])] : deps[x]);
  };
  const binding = (name) => {
    if (!(name in slots)) slots[name] = emit(trees.get(name));
    return slots[name];
  };
  for (const name of trees.keys()) binding(name);
  const pars = {};
  for (const name of trees.keys()) pars[name] = deps[slots[name]];
  return { n: op.length, op: Int32Array.from(op), a: Int32Array.from(a), b: Int32Array.from(b), value: Float64Array.from(value), slots, pars };
}

// values of all slots for the given parameters
function runTape(tape, pars, val) {
  const { op, a, b } = tape;
  for (let i = 0; i < tape.n; i++) {
    const x = val[a[i]];
    const y = val[b[i]];
    switch (op[i]) {
      case 0: val[i] = tape.value[i]; break;
      case 1: val[i] = pars[a[i]]; break;
      case 2: val[i] = x + y; break;
      case 3: val[i] = x - y; break;
      case 4: val[i] = x * y; break;
      case 5: val[i] = x / y; break;
      case 6: val[i] = Math.pow(x, y); break;
      case 7: val[i] = -x; break;
      case 8: val[i] = Math.sqrt(x); break;
      case 9: val[i] = Math.exp(x); break;
      case 10: val[i] = Math.log(x); break;
      case 11: val[i] = Math.sin(x); break;
      case 12: val[i] = Math.cos(x); break;
      case 13: val[i] = Math.tan(x); break;
      case 14: val[i] = Math.tanh(x); break;
      case 15: val[i] = Math.abs(x); break;
      case 16: val[i] = Math.asin(x); break;
      case 17: val[i] = Math.acos(x); break;
      case 18: val[i] = Math.atan(x); break;
      case 19: val[i] = Math.sinh(x); break;
      default: val[i] = Math.cosh(x);
    }
  }
}

// reverse sweep: adj holds the derivative of some quantity with respect to slots (zero where it does not
// depend on them directly); its derivative with respect to the parameters is added to grad. adj is used up.
function backTape(tape, val, adj, grad) {
  const { op, a, b } = tape;
  for (let i = tape.n - 1; i >= 0; i--) {
    const g = adj[i];
    if (g === 0) continue;
    const v = val[i];
    const x = val[a[i]];
    const y = val[b[i]];
    switch (op[i]) {
      case 0: break;
      case 1: grad[a[i]] += g; break;
      case 2: adj[a[i]] += g; adj[b[i]] += g; break;
      case 3: adj[a[i]] += g; adj[b[i]] -= g; break;
      case 4: adj[a[i]] += g * y; adj[b[i]] += g * x; break;
      case 5: adj[a[i]] += g / y; adj[b[i]] -= (g * v) / y; break;
      case 6:
        adj[a[i]] += g * y * Math.pow(x, y - 1);
        if (op[b[i]] !== 0) adj[b[i]] += g * v * Math.log(x);
        break;
      case 7: adj[a[i]] -= g; break;
      case 8: adj[a[i]] += g / (2 * v); break;
      case 9: adj[a[i]] += g * v; break;
      case 10: adj[a[i]] += g / x; break;
      case 11: adj[a[i]] += g * Math.cos(x); break;
      case 12: adj[a[i]] -= g * Math.sin(x); break;
      case 13: adj[a[i]] += g * (1 + v * v); break;
      case 14: adj[a[i]] += g * (1 - v * v); break;
      case 15: adj[a[i]] += x < 0 ? -g : g; break;
      case 16: adj[a[i]] += g / Math.sqrt(1 - x * x); break;
      case 17: adj[a[i]] -= g / Math.sqrt(1 - x * x); break;
      case 18: adj[a[i]] += g / (1 + x * x); break;
      case 19: adj[a[i]] += g * Math.cosh(x); break;
      default: adj[a[i]] += g * Math.sinh(x);
    }
  }
}

export { parseBindings, compileBindings, runTape, backTape };
