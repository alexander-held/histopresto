<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo.svg" alt="histopresto" width="420">
</picture>

**This is an R&D project: implementations and interfaces may change frequently.**

Fast fits of HistFactory models (pyhf workspaces), in JavaScript and in Python with a C kernel.
Both implementations use the same algorithm: sparse per-sample evaluation of the model, an analytic
gradient, a quasi-Newton minimiser with box bounds, and the covariance from finite differences of
the gradient. They agree with each other to rounding and are tested against pyhf.

**[Try it in the browser](https://alexander-held.github.io/histopresto/)**: load a pyhf workspace
and fit it.

| Layer | Where | What |
|---|---|---|
| Core | `js/core` (`@histopresto/core`), `python` (`histopresto`) | workspace parsing, model, likelihood and gradient, minimisation, fits, scans, ranking, CLs limits, significance, yields |
| Visualization | `js/viz` (`@histopresto/viz`) | SVG figures: pulls, correlations, ranking, data/MC, likelihood scans of one and two parameters with 68.3% and 95% regions, CLs scan |
| Website | `js/app` | static page that fits a workspace in the browser and shows all of the above |
| Tests | `tests` | both cores against reference numbers from pyhf, and against each other |

The core packages have no dependencies beyond numpy (Python) and nothing at all (JavaScript). The
visualization takes plain numbers and names, not objects of the core.

## Setup

Everything runs through [pixi](https://pixi.sh), which provides Python, Node.js and a C compiler:

```
pixi run test          # JavaScript and Python tests
pixi run build-app     # website into js/app/dist/ (open index.html, also works from file://)
pixi run serve         # website on http://localhost:8000, rebuilt on change
```

## Python

```
pip install ./python   # needs a C compiler and CMake at install time, nothing at run time
```

```python
import pathlib
import histopresto
from histopresto import infer

model = histopresto.Model(histopresto.parse_workspace(pathlib.Path("workspace.json").read_text()))
lik = model.likelihood(model.observed)
fit = infer.fit(model, lik)                  # fit.pars, fit.unc, fit.cov, fit.twice_nll, fit.status
limit = infer.limit(model, lik, fit)         # limit.observed, limit.expected
scan = infer.scan(model, lik, fit, model.poi_index, infer.linspace(0, 3, 25))   # scan.intervals: 68.3% and 95%
grid = infer.scan2d(model, lik, fit, [0, 1], infer.linspace(0, 3, 15), infer.linspace(-2, 2, 15))  # grid.twice_delta_nll[i][j]
asimov = model.likelihood(model.expected_data(model.init))   # fit Asimov instead of observed data
```

The kernel (`python/src/histopresto/kernel.c`) is built by scikit-build-core as a shared library and
loaded with ctypes. Linux and macOS only so far. In the pixi environment the package is installed in
editable mode; after changing `kernel.c` or `CMakeLists.txt`, rebuild with `pixi reinstall histopresto`.

## JavaScript

```js
import fs from "node:fs";
import { Model, infer, parseWorkspace } from "@histopresto/core";

const model = new Model(parseWorkspace(fs.readFileSync("workspace.json", "utf8")));
const lik = model.likelihood(model.observed);
const fit = await infer.fit(model, lik);               // fit.pars, fit.unc, fit.cov, fit.twiceNll, fit.status
const limit = await infer.limit(model, lik, fit);      // limit.observed, limit.expected
```

The packages are ES modules and are not published to npm yet; inside this repository they resolve
through npm workspaces.

## Expressions

Workspaces may scale samples by functions of parameters, in the format of
[pyhf PR 2717](https://github.com/scikit-hep/pyhf/pull/2717) (not merged, so the format can still
change): a `purefunc` modifier names a function that a top-level `bindings` list defines.

```json
"modifiers": [{"name": "k_ggF", "type": "purefunc"}],
...
"bindings": [{"names": ["k_ggF", "k_VBF"], "expression": "(kappa_g**2, sqrt(kappa_V) + k_ggF / 10)", "language": "sympy"}]
```

Symbols that are not bindings become unconstrained parameters (start 1, range 0 to 10), configured
like any other in the measurement. Bindings that no modifier uses are ignored and add no parameters
(pyhf adds parameters for them, which nothing depends on). The expressions are parsed here, without sympy; supported are
numbers, `pi`, `E`, `+ - * / **`, parentheses and `sqrt exp log ln sin cos tan tanh asin acos atan sinh cosh Abs`. Bindings may
refer to each other in any order as long as there is no cycle (pyhf only follows one level). The
gradient is exact, by a reverse sweep over the compiled expressions.

## Tests

`tests/reference/reference.json` holds numbers from pyhf for the workspaces in
`tests/reference/workspaces/`: parameter order and defaults, yields, likelihood, gradient, best fit,
uncertainties, correlations, CLs, limits and discovery significance. `tests/js` and `tests/python`
run the same 7370 checks against it. `tests/cross` checks that the two implementations take the same
number of iterations and agree to rounding.

Regenerating the reference (`pixi run reference`) takes about five minutes, nearly all of it pyhf on
the largest workspace. It skips the workspaces with expressions, which need pyhf from PR 2717 with
jax: `pixi run -e purefunc-ref reference-purefunc`.

## Known limitations

- Models with flat directions: when the covariance matrix is not positive definite the fit is
  reported as not converged, and uncertainties, ranking and bands are unreliable.
- `infer.fit` computes a Hessian at the starting point to seed the minimiser, which is discarded
  whenever it is not positive definite (the case on all larger models tried). It costs one gradient
  call per parameter.
- `infer.conditional` (used by scans, ranking, limits and significance) has no final Newton step, so
  on models with several hundred parameters it can stop slightly above the minimum.
- The dense BFGS update dominates above a few hundred parameters.
- Out of scope: patchsets, toy-based limits, XML/ROOT input. Bins have no physical edges.

## License

MIT, see `LICENSE`.
