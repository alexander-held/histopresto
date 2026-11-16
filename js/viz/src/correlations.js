// Correlation matrix of fit parameters.
import { figure } from "./plot.js";
import { fmt, short } from "./format.js";

// names: n parameter names; rho(i, j): correlation of parameters i and j
function drawCorrelations(container, names, rho) {
  const n = names.length;
  const longest = Math.max(...names.map((nm) => short(nm, 36).length)) + 4;
  const left = Math.min(300, 24 + 7 * longest);
  const cell = Math.max(12, Math.min(34, 640 / n));
  const size = cell * n;
  // drawn one to one: the figure is at least as wide as the layout needs to keep the cells square
  const W = Math.max(720, left + size + 20);
  container.style.maxWidth = `${W}px`;
  const fig = figure(container, { width: W, height: size + 60 });
  const a = fig.axes({
    box: [0, 0, W, size + 60], xlim: [0, n], ylim: [0, n], grid: false, margin: { left, right: W - left - size, top: 14, bottom: 46 },
    yticks: names.map((nm, i) => ({ v: n - 1 - i + 0.5, label: `${short(nm, 36)}  ${i + 1}` })),
    xticks: names.map((_, i) => ({ v: i + 0.5, label: n > 24 && i % 2 ? "" : String(i + 1) })),
    tip: (x, y) => {
      const i = n - 1 - Math.floor(y);
      const j = Math.floor(x);
      if (i < 0 || j < 0 || i >= n || j >= n) return null;
      return i === j ? names[i] : `${names[i]}\n${names[j]}\ncorrelation  ${fmt(rho(i, j), 3)}`;
    },
  });
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const r = rho(i, j);
      const pct = Math.round(100 * Math.min(Math.abs(r), 1));
      a.rect(j + 0.02, n - 1 - i + 0.02, j + 0.98, n - i - 0.02, { fill: `color-mix(in oklab, var(${r < 0 ? "--s1" : "--s8"}) ${pct}%, var(--mid))` });
      if (n <= 16 && i !== j) a.text(j + 0.5, n - 1 - i + 0.5, fmt(Math.abs(r) < 0.005 ? 0 : r, 2), { anchor: "middle", size: 10.5, color: Math.abs(r) > 0.6 ? "#fff" : "var(--ink)" });
    }
  }
  return fig.render();
}

export { drawCorrelations };
