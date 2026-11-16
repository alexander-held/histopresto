// Special functions and dense linear algebra on flat Float64Array matrices.

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];
function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// complementary error function: series for small arguments, continued fraction for large ones
function erfc(x) {
  if (x < 0) return 2 - erfc(-x);
  if (x < 2.5) {
    let term = x;
    let s = x;
    for (let n = 1; n < 100; n++) {
      term *= (-x * x) / n;
      const add = term / (2 * n + 1);
      s += add;
      if (Math.abs(add) < 1e-17 * Math.abs(s)) break;
    }
    return 1 - (2 / Math.sqrt(Math.PI)) * s;
  }
  let f = 0;
  for (let k = 80; k >= 1; k--) f = (k / 2) / (x + f);
  return Math.exp(-x * x) / (Math.sqrt(Math.PI) * (x + f));
}
const normSf = (x) => 0.5 * erfc(x / Math.SQRT2);
const normCdf = (x) => 0.5 * erfc(-x / Math.SQRT2);

function linspace(a, b, n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = n === 1 ? a : a + ((b - a) * i) / (n - 1);
  return out;
}

// Cholesky factor (lower triangle, in place on a copy) of a symmetric n x n matrix; null if not positive definite
function cholesky(A, n) {
  const L = new Float64Array(A);
  for (let j = 0; j < n; j++) {
    let d = L[j * n + j];
    for (let k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
    if (!(d > 0)) return null;
    d = Math.sqrt(d);
    L[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let s = L[i * n + j];
      const ri = i * n;
      const rj = j * n;
      for (let k = 0; k < j; k++) s -= L[ri + k] * L[rj + k];
      L[ri + j] = s / d;
    }
  }
  return L;
}
// solve A x = b given the Cholesky factor of A; b is overwritten with x
function cholSolve(L, n, b) {
  for (let i = 0; i < n; i++) {
    let s = b[i];
    const ri = i * n;
    for (let k = 0; k < i; k++) s -= L[ri + k] * b[k];
    b[i] = s / L[ri + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * b[k];
    b[i] = s / L[i * n + i];
  }
  return b;
}
// inverse of a symmetric positive definite matrix from its Cholesky factor
function cholInverse(L, n) {
  // invert the triangular factor, then form inv(L)^T inv(L)
  const T = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    T[j * n + j] = 1 / L[j * n + j];
    for (let i = j + 1; i < n; i++) {
      let s = 0;
      const ri = i * n;
      for (let k = j; k < i; k++) s -= L[ri + k] * T[k * n + j];
      T[ri + j] = s / L[ri + i];
    }
  }
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = i; k < n; k++) s += T[k * n + i] * T[k * n + j];
      out[i * n + j] = out[j * n + i] = s;
    }
  }
  return out;
}
// general inverse by Gauss-Jordan elimination with partial pivoting; null if singular
function inverse(A, n) {
  const M = new Float64Array(A);
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[piv * n + c])) piv = r;
    if (Math.abs(M[piv * n + c]) < 1e-300) return null;
    if (piv !== c) {
      for (let k = 0; k < n; k++) {
        let t = M[c * n + k];
        M[c * n + k] = M[piv * n + k];
        M[piv * n + k] = t;
        t = I[c * n + k];
        I[c * n + k] = I[piv * n + k];
        I[piv * n + k] = t;
      }
    }
    const d = 1 / M[c * n + c];
    for (let k = 0; k < n; k++) {
      M[c * n + k] *= d;
      I[c * n + k] *= d;
    }
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r * n + c];
      if (f === 0) continue;
      for (let k = 0; k < n; k++) {
        M[r * n + k] -= f * M[c * n + k];
        I[r * n + k] -= f * I[c * n + k];
      }
    }
  }
  return I;
}

// minimum of y(x) and the points around it where y has risen by `level`
function interval(x, y, level = 1) {
  let i = 0;
  for (let k = 1; k < y.length; k++) if (y[k] < y[i]) i = k;
  const y0 = y[i];
  const cross = (a, b) => x[a] + ((level - (y[a] - y0)) / (y[b] - y[a])) * (x[b] - x[a]);
  let lo = NaN;
  let hi = NaN;
  for (let k = i; k > 0; k--) {
    if (y[k - 1] - y0 >= level) {
      lo = cross(k, k - 1);
      break;
    }
  }
  for (let k = i; k < x.length - 1; k++) {
    if (y[k + 1] - y0 >= level) {
      hi = cross(k, k + 1);
      break;
    }
  }
  return { min: x[i], lo, hi, index: i };
}

export { lgamma, erfc, normSf, normCdf, linspace, cholesky, cholSolve, cholInverse, inverse, interval };
