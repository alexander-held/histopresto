// Figures for HistFactory fit results, drawn as SVG into a DOM element. They take plain numbers and
// names, not objects of @histopresto/core. Colours are CSS custom properties that the page defines (see ../README.md).
export * as plot from "./plot.js";
export { layoutKey } from "./plot.js";
export { esc, fmt, num, yieldFmt, short } from "./format.js";
export { pull, drawPulls } from "./pulls.js";
export { drawCorrelations } from "./correlations.js";
export { drawRanking } from "./ranking.js";
export { drawYields } from "./yields.js";
export { drawScan } from "./scan.js";
export { drawLimit } from "./limits.js";
