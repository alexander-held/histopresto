// Number and text formatting shared by the figures and by pages showing fit results.
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const minus = (s) => s.replace(/^-/, "−");
const fmt = (v, d = 2) => (Number.isFinite(v) ? minus(v.toFixed(d)) : "–");
// number with a sensible precision whatever its size
const num = (v, sig = 3) => {
  if (!Number.isFinite(v)) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return minus(v.toExponential(sig - 1));
  return minus(v.toFixed(Math.max(0, sig - 1 - Math.floor(Math.log10(a)))));
};
const yieldFmt = (v) => (!Number.isFinite(v) ? "–" : Math.abs(v) >= 1000 || Number.isInteger(v) ? minus(v.toFixed(0)) : Math.abs(v) >= 10 ? minus(v.toFixed(1)) : minus(v.toFixed(2)));
// long parameter names keep their start and end
const short = (name, max = 44) => (name.length > max ? `${name.slice(0, max - 12)}…${name.slice(-10)}` : name);

export { esc, fmt, num, yieldFmt, short };
