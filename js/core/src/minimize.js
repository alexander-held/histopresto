// Quasi-Newton minimisation with box bounds, and second derivatives from central differences of the gradient.
import { inverse } from "./numerics.js";

// fg(x, grad) returns the function value and fills the gradient (both over all parameters).
// free: indices of the parameters to vary. options: C (inverse Hessian guess over the free
// parameters, flat n x n), diag (fallback diagonal of the inverse Hessian), tol (estimated
// distance to the minimum at which to stop), maxIter.
function minimize(fg, x0, lower, upper, free, options = {}) {
  const N = x0.length;
  const n = free.length;
  const tol = options.tol || 1e-7;
  const maxIter = options.maxIter || 200 + 20 * n;
  const diag = options.diag;
  let x = Float64Array.from(x0);
  for (let i = 0; i < n; i++) {
    const p = free[i];
    x[p] = Math.min(Math.max(x[p], lower[p]), upper[p]);
  }
  let g = new Float64Array(N);
  let f = fg(x, g);
  let xn = new Float64Array(N);
  let gn = new Float64Array(N);
  const C = new Float64Array(n * n);
  const setDiagonal = () => {
    C.fill(0);
    for (let i = 0; i < n; i++) C[i * n + i] = diag ? diag[i] : 1;
  };
  if (options.C) C.set(options.C);
  else setDiagonal();

  const gf = new Float64Array(n);
  const u = new Float64Array(n);
  const s = new Float64Array(n);
  const y = new Float64Array(n);
  const Cy = new Float64Array(n);
  const active = new Uint8Array(n);
  let edm = Infinity;
  let converged = false;
  let resets = 0;
  let stalled = 0;
  let iter = 0;
  if (!Number.isFinite(f)) return { x, f, g, C, edm, converged: false, iterations: 0, active };

  for (; iter < maxIter && n > 0; iter++) {
    // an infinite gradient pointing into the allowed range (the square root of a parameter at zero, say)
    // gives no usable direction: step off the bound
    let moved = false;
    for (let i = 0; i < n; i++) {
      const p = free[i];
      if (x[p] <= lower[p] && g[p] === -Infinity) x[p] = lower[p] + 1e-6 * (upper[p] - lower[p]);
      else if (x[p] >= upper[p] && g[p] === Infinity) x[p] = upper[p] - 1e-6 * (upper[p] - lower[p]);
      else continue;
      moved = true;
    }
    if (moved) {
      f = fg(x, g);
      if (!Number.isFinite(f)) break;
    }
    // parameters at a bound with the gradient pushing outwards do not move
    const act = [];
    for (let i = 0; i < n; i++) {
      const p = free[i];
      active[i] = (x[p] <= lower[p] && g[p] > 0) || (x[p] >= upper[p] && g[p] < 0) ? 1 : 0;
      gf[i] = active[i] ? 0 : g[p];
      if (active[i]) act.push(i);
    }
    for (let i = 0; i < n; i++) {
      let t = 0;
      const r = i * n;
      for (let j = 0; j < n; j++) t += C[r + j] * gf[j];
      u[i] = t;
    }
    if (act.length) {
      // remove the active directions: u <- u - C[:,A] inv(C[A,A]) u[A]
      const k = act.length;
      const CAA = new Float64Array(k * k);
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) CAA[a * k + b] = C[act[a] * n + act[b]];
      const inv = inverse(CAA, k);
      if (inv) {
        const z = new Float64Array(k);
        for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) z[a] += inv[a * k + b] * u[act[b]];
        for (let i = 0; i < n; i++) {
          let t = 0;
          for (let a = 0; a < k; a++) t += C[i * n + act[a]] * z[a];
          u[i] -= t;
        }
      }
      for (const a of act) u[a] = 0;
    }
    let slope = 0;
    for (let i = 0; i < n; i++) slope -= gf[i] * u[i];
    edm = -0.5 * slope;
    if (edm < tol && edm >= 0) {
      converged = true;
      break;
    }
    if (!(slope < 0)) {
      if (resets++ > 3) break;
      setDiagonal();
      continue;
    }

    // backtracking line search along -u, projected onto the bounds
    let t = 1;
    let fn = Infinity;
    let accepted = false;
    for (let trial = 0; trial < 40; trial++) {
      xn.set(x);
      let predicted = 0;
      for (let i = 0; i < n; i++) {
        const p = free[i];
        const v = Math.min(Math.max(x[p] - t * u[i], lower[p]), upper[p]);
        xn[p] = v;
        if (v !== x[p]) predicted += g[p] * (v - x[p]);
      }
      fn = fg(xn, gn);
      if (Number.isFinite(fn) && fn <= f + 1e-4 * predicted) {
        accepted = true;
        break;
      }
      const quad = Number.isFinite(fn) ? (-slope * t * t) / (2 * (fn - f - slope * t)) : 0;
      t = Math.min(Math.max(quad, 0.1 * t), 0.5 * t);
    }
    if (!accepted) {
      if (resets++ > 3) break;
      setDiagonal();
      continue;
    }

    let sy = 0;
    let ss = 0;
    let yy = 0;
    for (let i = 0; i < n; i++) {
      const p = free[i];
      s[i] = xn[p] - x[p];
      y[i] = gn[p] - g[p];
      sy += s[i] * y[i];
      ss += s[i] * s[i];
      yy += y[i] * y[i];
    }
    if (sy > 1e-10 * Math.sqrt(ss * yy)) {
      let yCy = 0;
      for (let i = 0; i < n; i++) {
        let v = 0;
        const r = i * n;
        for (let j = 0; j < n; j++) v += C[r + j] * y[j];
        Cy[i] = v;
        yCy += v * y[i];
      }
      const a = (sy + yCy) / (sy * sy);
      for (let i = 0; i < n; i++) {
        const r = i * n;
        const si = s[i];
        const ci = Cy[i];
        for (let j = 0; j < n; j++) C[r + j] += a * si * s[j] - (ci * s[j] + si * Cy[j]) / sy;
      }
    }
    stalled = f - fn < 1e-13 * (Math.abs(f) + 1) ? stalled + 1 : 0;
    [x, xn] = [xn, x];
    [g, gn] = [gn, g];
    f = fn;
    if (stalled >= 3) break;
  }
  return { x, f, g, C, edm, converged, iterations: iter, active };
}

// second derivatives of f with respect to the parameters in idx, from central differences of the gradient
// (one-sided where f is not defined on the other side)
function hessian(fg, x, idx, steps) {
  const N = x.length;
  const k = idx.length;
  const H = new Float64Array(k * k);
  const xp = Float64Array.from(x);
  const g0 = new Float64Array(N);
  const gp = new Float64Array(N);
  const gm = new Float64Array(N);
  for (let a = 0; a < k; a++) {
    const p = idx[a];
    const h = steps[a];
    xp[p] = x[p] + h;
    const up = Number.isFinite(fg(xp, gp));
    xp[p] = x[p] - h;
    const down = Number.isFinite(fg(xp, gm));
    xp[p] = x[p];
    if (!up || !down) fg(xp, g0);
    for (let b = 0; b < k; b++) {
      const q = idx[b];
      if (up && down) H[a * k + b] = (gp[q] - gm[q]) / (2 * h);
      else if (up) H[a * k + b] = (gp[q] - g0[q]) / h;
      else if (down) H[a * k + b] = (g0[q] - gm[q]) / h;
      else H[a * k + b] = NaN;
    }
  }
  for (let a = 0; a < k; a++) {
    for (let b = a + 1; b < k; b++) H[a * k + b] = H[b * k + a] = 0.5 * (H[a * k + b] + H[b * k + a]);
  }
  return H;
}

export { minimize, hessian };
