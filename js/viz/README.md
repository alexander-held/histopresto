# @histopresto/viz

Figures for HistFactory fit results, drawn as SVG into a DOM element. The functions take plain
numbers and names, so they work with results from `@histopresto/core`, from the Python package, or from any
other fitter.

| Function | Figure |
|---|---|
| `drawPulls(el, rows)` | pulls and constraints of nuisance parameters |
| `drawCorrelations(el, names, rho)` | correlation matrix |
| `drawRanking(el, rows, poiName)` | impact on the parameter of interest, with pulls |
| `drawYields(el, channel, layers, options)` | data and stacked expectation of one channel, with ratio |
| `drawScan(el, scan, options)` | profile likelihood scan |
| `drawLimit(el, limit, options)` | CLs against the parameter of interest, with expected bands |

The shape of each argument is described above the function in `src/`. `pull(param, value, unc)`
computes the pull of a constrained parameter; `fmt`, `num`, `yieldFmt`, `short` and `esc` format
numbers and names the way the figures do. `plot` is the underlying plotting helper.

```js
import { drawPulls, pull } from "@histopresto/viz";

const rows = parameters.map((p) => ({ name: p.name, value: p.value, unc: p.unc, ...pull(p, p.value, p.unc) }));
drawPulls(document.getElementById("pulls"), rows);
```

## Colours

Figures refer to CSS custom properties, so they follow the page (including a dark scheme). The page
has to define:

`--ink`, `--ink-2`, `--grey`, `--axis`, `--grid`, `--plot-bg`, `--mid`, `--band-1`, `--band-2`,
`--rank-up`, `--rank-down`, `--s1`, `--s2`, `--s8`

`js/app/style.css` has a light and a dark set of values, and the styles for `.figure` and the
`.readout` shown under the pointer.
