// Numerical kernel of the HistFactory fitter: expected yields, -2 ln L with its gradient,
// quasi-Newton minimisation with box bounds and second derivatives from gradient differences.
// The loops follow js/core/src/model.js, expression.js and minimize.js statement by statement.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Layout must match the ctypes Structure in _kernel.py. All memory is owned by the caller.
typedef struct {
  int32_t n_bins, n_pars, n_blocks, n_cells, n_gauss, n_pois, n_slots;
  // one block per defined (channel, sample)
  const int32_t *blk_offset, *blk_n, *blk_cell;
  const double *nominal;                    // per cell
  const int32_t *scal_start, *scal_count;   // per block, into scal_par
  const int32_t *scal_par;
  const int32_t *ns_start, *ns_count;       // per block, into the ns_* tables
  const int32_t *ns_par;
  const double *ns_lnhi, *ns_lnlo, *ns_coef; // ns_coef: 6 per entry
  const int32_t *hs_start, *hs_count;       // per block, into the hs_* tables
  const int32_t *hs_par, *hs_data;          // hs_data: offset of the entry's bins in hs_up / hs_dn
  const double *hs_up, *hs_dn;
  const int32_t *gam_start, *gam_count;     // per block, into gam_off
  const int32_t *gam_off;                   // offset of the entry's bins in gam_idx
  const int32_t *gam_idx;                   // parameter index per bin, -1 where the modifier is absent
  // constraint terms
  const int32_t *gauss_idx, *gauss_aux;
  const double *gauss_sigma;
  const int32_t *pois_idx, *pois_aux;
  const double *pois_tau;
  // scratch space filled by hf_evaluate
  double *nu, *cell_nu, *cell_base, *cell_gamma, *block_factor, *cell_weight, *ns_val, *ns_der;
  // purefunc modifiers: a tape of operations (see expression.py) and the slots that multiply each block
  const int32_t *fn_start, *fn_count;       // per block, into fn_slot
  const int32_t *fn_slot;
  const int32_t *tape_op, *tape_a, *tape_b; // per slot
  const double *tape_value;                 // per slot, used by constants
  double *slot_val, *slot_adj;              // scratch space
} hf_model;

static const double LN2PI = 1.8378770664093453;

// Math.min / Math.max of JavaScript: NaN if either argument is NaN
static inline double jmin(double a, double b) { return (isnan(a) || isnan(b)) ? NAN : (a < b ? a : b); }
static inline double jmax(double a, double b) { return (isnan(a) || isnan(b)) ? NAN : (a > b ? a : b); }
static inline double clamp(double v, double lo, double hi) { return jmin(jmax(v, lo), hi); }

enum { OP_CONST, OP_PAR, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW, OP_NEG, OP_SQRT, OP_EXP, OP_LOG, OP_SIN, OP_COS, OP_TAN, OP_TANH, OP_ABS,
       OP_ASIN, OP_ACOS, OP_ATAN, OP_SINH, OP_COSH };

// values of all slots of the tape for the given parameters
static void run_tape(const hf_model *m, const double *pars) {
  const int32_t *op = m->tape_op, *a = m->tape_a, *b = m->tape_b;
  double *val = m->slot_val;
  for (int i = 0; i < m->n_slots; i++) {
    switch (op[i]) {
      case OP_CONST: val[i] = m->tape_value[i]; break;
      case OP_PAR: val[i] = pars[a[i]]; break;
      case OP_ADD: val[i] = val[a[i]] + val[b[i]]; break;
      case OP_SUB: val[i] = val[a[i]] - val[b[i]]; break;
      case OP_MUL: val[i] = val[a[i]] * val[b[i]]; break;
      case OP_DIV: val[i] = val[a[i]] / val[b[i]]; break;
      case OP_POW: val[i] = pow(val[a[i]], val[b[i]]); break;
      case OP_NEG: val[i] = -val[a[i]]; break;
      case OP_SQRT: val[i] = sqrt(val[a[i]]); break;
      case OP_EXP: val[i] = exp(val[a[i]]); break;
      case OP_LOG: val[i] = log(val[a[i]]); break;
      case OP_SIN: val[i] = sin(val[a[i]]); break;
      case OP_COS: val[i] = cos(val[a[i]]); break;
      case OP_TAN: val[i] = tan(val[a[i]]); break;
      case OP_TANH: val[i] = tanh(val[a[i]]); break;
      case OP_ABS: val[i] = fabs(val[a[i]]); break;
      case OP_ASIN: val[i] = asin(val[a[i]]); break;
      case OP_ACOS: val[i] = acos(val[a[i]]); break;
      case OP_ATAN: val[i] = atan(val[a[i]]); break;
      case OP_SINH: val[i] = sinh(val[a[i]]); break;
      default: val[i] = cosh(val[a[i]]);
    }
  }
}

// reverse sweep: slot_adj holds the derivative of some quantity with respect to slots; its derivative
// with respect to the parameters is added to grad
static void back_tape(const hf_model *m, double *grad) {
  const int32_t *op = m->tape_op, *a = m->tape_a, *b = m->tape_b;
  const double *val = m->slot_val;
  double *adj = m->slot_adj;
  for (int i = m->n_slots - 1; i >= 0; i--) {
    const double g = adj[i];
    if (g == 0) continue;
    const double v = val[i];
    switch (op[i]) {
      case OP_CONST: break;
      case OP_PAR: grad[a[i]] += g; break;
      case OP_ADD: adj[a[i]] += g; adj[b[i]] += g; break;
      case OP_SUB: adj[a[i]] += g; adj[b[i]] -= g; break;
      case OP_MUL: { const double x = val[a[i]], y = val[b[i]]; adj[a[i]] += g * y; adj[b[i]] += g * x; break; }
      case OP_DIV: { const double y = val[b[i]]; adj[a[i]] += g / y; adj[b[i]] -= (g * v) / y; break; }
      case OP_POW: {
        const double x = val[a[i]], y = val[b[i]];
        adj[a[i]] += g * y * pow(x, y - 1);
        if (op[b[i]] != OP_CONST) adj[b[i]] += g * v * log(x);
        break;
      }
      case OP_NEG: adj[a[i]] -= g; break;
      case OP_SQRT: adj[a[i]] += g / (2 * v); break;
      case OP_EXP: adj[a[i]] += g * v; break;
      case OP_LOG: adj[a[i]] += g / val[a[i]]; break;
      case OP_SIN: adj[a[i]] += g * cos(val[a[i]]); break;
      case OP_COS: adj[a[i]] -= g * sin(val[a[i]]); break;
      case OP_TAN: adj[a[i]] += g * (1 + v * v); break;
      case OP_TANH: adj[a[i]] += g * (1 - v * v); break;
      case OP_ABS: adj[a[i]] += val[a[i]] < 0 ? -g : g; break;
      case OP_ASIN: { const double x = val[a[i]]; adj[a[i]] += g / sqrt(1 - x * x); break; }
      case OP_ACOS: { const double x = val[a[i]]; adj[a[i]] -= g / sqrt(1 - x * x); break; }
      case OP_ATAN: { const double x = val[a[i]]; adj[a[i]] += g / (1 + x * x); break; }
      case OP_SINH: adj[a[i]] += g * cosh(val[a[i]]); break;
      default: adj[a[i]] += g * sinh(val[a[i]]);
    }
  }
}

// expected events for the given parameters; results are left in m->nu and m->cell_nu
void hf_evaluate(hf_model *m, const double *pars) {
  double *nu = m->nu;
  memset(nu, 0, sizeof(double) * (size_t)m->n_bins);
  run_tape(m, pars);
  for (int i = 0; i < m->n_blocks; i++) {
    double F = 1;
    for (int k = m->scal_start[i], e = k + m->scal_count[i]; k < e; k++) F *= pars[m->scal_par[k]];
    for (int k = m->fn_start[i], e = k + m->fn_count[i]; k < e; k++) F *= m->slot_val[m->fn_slot[k]];
    for (int k = m->ns_start[i], e = k + m->ns_count[i]; k < e; k++) {
      const double a = pars[m->ns_par[k]];
      double v, d;
      if (a >= 1) {
        v = exp(a * m->ns_lnhi[k]);
        d = m->ns_lnhi[k] * v;
      } else if (a <= -1) {
        v = exp(-a * m->ns_lnlo[k]);
        d = -m->ns_lnlo[k] * v;
      } else {
        const double *c = m->ns_coef + 6 * k;
        v = 1 + a * (c[0] + a * (c[1] + a * (c[2] + a * (c[3] + a * (c[4] + a * c[5])))));
        d = c[0] + a * (2 * c[1] + a * (3 * c[2] + a * (4 * c[3] + a * (5 * c[4] + a * 6 * c[5]))));
      }
      m->ns_val[k] = v;
      m->ns_der[k] = d;
      F *= v;
    }
    m->block_factor[i] = F;
    const int n = m->blk_n[i];
    const int c0 = m->blk_cell[i];
    double *base = m->cell_base + c0;
    memcpy(base, m->nominal + c0, sizeof(double) * (size_t)n);
    for (int k = m->hs_start[i], e = k + m->hs_count[i]; k < e; k++) {
      const double a = pars[m->hs_par[k]];
      const double *up = m->hs_up + m->hs_data[k];
      const double *dn = m->hs_dn + m->hs_data[k];
      if (a > 1) {
        for (int b = 0; b < n; b++) base[b] += up[b] * a;
      } else if (a < -1) {
        for (int b = 0; b < n; b++) base[b] += dn[b] * a;
      } else {
        const double poly = a * a * (15 + a * a * (-10 + a * a * 3));
        for (int b = 0; b < n; b++) base[b] += 0.5 * (up[b] + dn[b]) * a + 0.0625 * (up[b] - dn[b]) * poly;
      }
    }
    const int g0 = m->gam_start[i], g1 = g0 + m->gam_count[i];
    double *out = nu + m->blk_offset[i];
    for (int b = 0; b < n; b++) {
      double g = 1;
      for (int k = g0; k < g1; k++) {
        const int idx = m->gam_idx[m->gam_off[k] + b];
        if (idx >= 0) g *= pars[idx];
      }
      m->cell_gamma[c0 + b] = g;
      const double v = base[b] * F * g;
      m->cell_nu[c0 + b] = v;
      out[b] += v;
    }
  }
}

// adds to grad the derivative of sum_cells cell_weight * cell_nu with respect to the parameters, for the
// listed blocks (all if blocks is NULL); hf_evaluate must have been called with the same parameters
void hf_accumulate(hf_model *m, const double *pars, double *grad, const int32_t *blocks, int n_list) {
  const double *cell_nu = m->cell_nu, *cell_base = m->cell_base, *cell_gamma = m->cell_gamma, *w = m->cell_weight;
  const double *slot_val = m->slot_val;
  double *slot_adj = m->slot_adj;
  memset(slot_adj, 0, sizeof(double) * (size_t)m->n_slots);
  const int count = blocks ? n_list : m->n_blocks;
  for (int q = 0; q < count; q++) {
    const int i = blocks ? blocks[q] : q;
    const int n = m->blk_n[i];
    const int c0 = m->blk_cell[i];
    const double F = m->block_factor[i];
    double T = 0, U = 0;
    int any = 0;
    for (int b = 0; b < n; b++) {
      const double wb = w[c0 + b];
      if (wb == 0) continue;
      any = 1;
      T += wb * cell_nu[c0 + b];
      U += wb * cell_base[c0 + b] * cell_gamma[c0 + b];
    }
    if (!any) continue;
    const int s0 = m->scal_start[i], s1 = s0 + m->scal_count[i];
    const int n0 = m->ns_start[i], n1 = n0 + m->ns_count[i];
    const int f0 = m->fn_start[i], f1 = f0 + m->fn_count[i];
    for (int k = s0; k < s1; k++) {
      const int p = m->scal_par[k];
      if (pars[p] != 0) grad[p] += T / pars[p];
      else {
        double rest = 1;
        for (int j = s0; j < s1; j++) if (j != k) rest *= pars[m->scal_par[j]];
        for (int j = f0; j < f1; j++) rest *= slot_val[m->fn_slot[j]];
        for (int j = n0; j < n1; j++) rest *= m->ns_val[j];
        grad[p] += U * rest;
      }
    }
    for (int k = f0; k < f1; k++) {
      const int s = m->fn_slot[k];
      if (slot_val[s] != 0) slot_adj[s] += T / slot_val[s];
      else {
        double rest = 1;
        for (int j = s0; j < s1; j++) rest *= pars[m->scal_par[j]];
        for (int j = f0; j < f1; j++) if (j != k) rest *= slot_val[m->fn_slot[j]];
        for (int j = n0; j < n1; j++) rest *= m->ns_val[j];
        slot_adj[s] += U * rest;
      }
    }
    for (int k = n0; k < n1; k++) grad[m->ns_par[k]] += (T * m->ns_der[k]) / m->ns_val[k];
    for (int k = m->hs_start[i], e = k + m->hs_count[i]; k < e; k++) {
      const double a = pars[m->hs_par[k]];
      const double *up = m->hs_up + m->hs_data[k];
      const double *dn = m->hs_dn + m->hs_data[k];
      double s = 0;
      if (a > 1) {
        for (int b = 0; b < n; b++) s += w[c0 + b] * cell_gamma[c0 + b] * up[b];
      } else if (a < -1) {
        for (int b = 0; b < n; b++) s += w[c0 + b] * cell_gamma[c0 + b] * dn[b];
      } else {
        const double dpoly = a * (30 + a * a * (-40 + a * a * 18));
        for (int b = 0; b < n; b++) s += w[c0 + b] * cell_gamma[c0 + b] * (0.5 * (up[b] + dn[b]) + 0.0625 * (up[b] - dn[b]) * dpoly);
      }
      grad[m->hs_par[k]] += F * s;
    }
    const int g0 = m->gam_start[i], g1 = g0 + m->gam_count[i];
    for (int k = g0; k < g1; k++) {
      const int32_t *list = m->gam_idx + m->gam_off[k];
      for (int b = 0; b < n; b++) {
        const int idx = list[b];
        const double wb = w[c0 + b];
        if (idx < 0 || wb == 0) continue;
        if (pars[idx] != 0) grad[idx] += (wb * cell_nu[c0 + b]) / pars[idx];
        else {
          double rest = 1;
          for (int j = g0; j < g1; j++) {
            const int other = m->gam_idx[m->gam_off[j] + b];
            if (j != k && other >= 0) rest *= pars[other];
          }
          grad[idx] += wb * cell_base[c0 + b] * F * rest;
        }
      }
    }
  }
  back_tape(m, grad);
}

// parameter-independent part of -2 ln L for a data vector (main data followed by auxiliary data)
double hf_constant(const hf_model *m, const double *data) {
  const int B = m->n_bins;
  double constant = 0;
  for (int b = 0; b < B; b++) constant += 2 * lgamma(data[b] + 1);
  for (int k = 0; k < m->n_gauss; k++) constant += 2 * log(m->gauss_sigma[k]) + LN2PI;
  for (int k = 0; k < m->n_pois; k++) constant += 2 * lgamma(data[B + m->pois_aux[k]] + 1);
  return constant;
}

// -2 ln L; fills grad unless it is NULL. Returns infinity where the model is not defined (grad untouched).
double hf_nll(hf_model *m, const double *data, double constant, const double *pars, double *grad) {
  const int B = m->n_bins;
  hf_evaluate(m, pars);
  const double *nu = m->nu;
  double t = constant;
  for (int b = 0; b < B; b++) {
    const double v = nu[b];
    if (v > 0) t += 2 * (v - data[b] * log(v));
    else if (!(v == 0 && data[b] == 0)) return INFINITY;
  }
  for (int k = 0; k < m->n_gauss; k++) {
    const double z = (data[B + m->gauss_aux[k]] - pars[m->gauss_idx[k]]) / m->gauss_sigma[k];
    t += z * z;
  }
  for (int k = 0; k < m->n_pois; k++) {
    const double a = data[B + m->pois_aux[k]];
    const double rate = pars[m->pois_idx[k]] * m->pois_tau[k];
    if (rate > 0) t += 2 * (rate - a * log(rate));
    else if (!(rate == 0 && a == 0)) return INFINITY;
  }
  if (!isfinite(t)) return INFINITY;
  if (!grad) return t;

  memset(grad, 0, sizeof(double) * (size_t)m->n_pars);
  for (int i = 0; i < m->n_blocks; i++) {
    const int n = m->blk_n[i];
    const double *v = nu + m->blk_offset[i];
    const double *d = data + m->blk_offset[i];
    double *w = m->cell_weight + m->blk_cell[i];
    for (int b = 0; b < n; b++) w[b] = v[b] > 0 ? 2 * (1 - d[b] / v[b]) : 2;
  }
  hf_accumulate(m, pars, grad, NULL, 0);
  for (int k = 0; k < m->n_gauss; k++) {
    const int p = m->gauss_idx[k];
    grad[p] += (2 * (pars[p] - data[B + m->gauss_aux[k]])) / (m->gauss_sigma[k] * m->gauss_sigma[k]);
  }
  for (int k = 0; k < m->n_pois; k++) {
    const int p = m->pois_idx[k];
    const double a = data[B + m->pois_aux[k]];
    grad[p] += 2 * (m->pois_tau[k] - (pars[p] > 0 ? a / pars[p] : 0));
  }
  return t;
}

// general inverse by Gauss-Jordan elimination with partial pivoting; A is destroyed, returns 0 if singular
static int invert(double *M, double *I, int n) {
  memset(I, 0, sizeof(double) * (size_t)n * (size_t)n);
  for (int i = 0; i < n; i++) I[i * n + i] = 1;
  for (int c = 0; c < n; c++) {
    int piv = c;
    for (int r = c + 1; r < n; r++) if (fabs(M[r * n + c]) > fabs(M[piv * n + c])) piv = r;
    if (fabs(M[piv * n + c]) < 1e-300) return 0;
    if (piv != c) {
      for (int k = 0; k < n; k++) {
        double t = M[c * n + k];
        M[c * n + k] = M[piv * n + k];
        M[piv * n + k] = t;
        t = I[c * n + k];
        I[c * n + k] = I[piv * n + k];
        I[piv * n + k] = t;
      }
    }
    const double d = 1 / M[c * n + c];
    for (int k = 0; k < n; k++) {
      M[c * n + k] *= d;
      I[c * n + k] *= d;
    }
    for (int r = 0; r < n; r++) {
      if (r == c) continue;
      const double f = M[r * n + c];
      if (f == 0) continue;
      for (int k = 0; k < n; k++) {
        M[r * n + k] -= f * M[c * n + k];
        I[r * n + k] -= f * I[c * n + k];
      }
    }
  }
  return 1;
}

// Minimises -2 ln L over the parameters listed in free (n of them); the others stay at x0.
// C0: inverse Hessian guess over the free parameters (n x n) or NULL; diag: fallback diagonal or NULL.
// tol: estimated distance to the minimum at which to stop; max_iter <= 0 selects 200 + 20 n.
// Outputs: x, g (all parameters), C (n x n), active (n), f, edm, iterations. Returns 1 if converged.
int hf_minimize(hf_model *m, const double *data, double constant, const double *x0, const double *lower,
                const double *upper, const int32_t *free_idx, int n, const double *C0, const double *diag,
                double tol, int max_iter, double *x_out, double *g_out, double *C, uint8_t *active,
                double *f_out, double *edm_out, int32_t *iter_out) {
  const int N = m->n_pars;
  const size_t nn = (size_t)n * (size_t)n;
  if (max_iter <= 0) max_iter = 200 + 20 * n;
  double *buf = (double *)malloc(sizeof(double) * (4 * (size_t)N + 5 * (size_t)n + 1));
  double *x = buf, *g = x + N, *xn = g + N, *gn = xn + N;
  double *gf = gn + N, *u = gf + n, *s = u + n, *y = s + n, *Cy = y + n;
  int *act = (int *)malloc(sizeof(int) * ((size_t)n + 1));
  double *CAA = NULL, *inv = NULL, *z = NULL;
  int cap = 0;

  memcpy(x, x0, sizeof(double) * (size_t)N);
  for (int i = 0; i < n; i++) {
    const int p = free_idx[i];
    x[p] = clamp(x[p], lower[p], upper[p]);
  }
  memset(g, 0, sizeof(double) * (size_t)N);
  memset(xn, 0, sizeof(double) * (size_t)N);
  memset(gn, 0, sizeof(double) * (size_t)N);
  memset(active, 0, (size_t)n);
  double f = hf_nll(m, data, constant, x, g);
#define SET_DIAGONAL()                                                  \
  do {                                                                  \
    memset(C, 0, sizeof(double) * nn);                                  \
    for (int i_ = 0; i_ < n; i_++) C[(size_t)i_ * n + i_] = diag ? diag[i_] : 1; \
  } while (0)
  if (C0) memcpy(C, C0, sizeof(double) * nn);
  else SET_DIAGONAL();

  double edm = INFINITY;
  int converged = 0, resets = 0, stalled = 0, iter = 0;
  if (isfinite(f)) {
    for (; iter < max_iter && n > 0; iter++) {
      // an infinite gradient pointing into the allowed range (the square root of a parameter at zero, say)
      // gives no usable direction: step off the bound
      int moved = 0;
      for (int i = 0; i < n; i++) {
        const int p = free_idx[i];
        if (x[p] <= lower[p] && g[p] == -INFINITY) x[p] = lower[p] + 1e-6 * (upper[p] - lower[p]);
        else if (x[p] >= upper[p] && g[p] == INFINITY) x[p] = upper[p] - 1e-6 * (upper[p] - lower[p]);
        else continue;
        moved = 1;
      }
      if (moved) {
        f = hf_nll(m, data, constant, x, g);
        if (!isfinite(f)) break;
      }
      // parameters at a bound with the gradient pushing outwards do not move
      int k = 0;
      for (int i = 0; i < n; i++) {
        const int p = free_idx[i];
        active[i] = ((x[p] <= lower[p] && g[p] > 0) || (x[p] >= upper[p] && g[p] < 0)) ? 1 : 0;
        gf[i] = active[i] ? 0 : g[p];
        if (active[i]) act[k++] = i;
      }
      for (int i = 0; i < n; i++) {
        double t = 0;
        const double *row = C + (size_t)i * n;
        for (int j = 0; j < n; j++) t += row[j] * gf[j];
        u[i] = t;
      }
      if (k) {
        // remove the active directions: u <- u - C[:,A] inv(C[A,A]) u[A]
        if (k > cap) {
          cap = k;
          CAA = (double *)realloc(CAA, sizeof(double) * (size_t)k * k);
          inv = (double *)realloc(inv, sizeof(double) * (size_t)k * k);
          z = (double *)realloc(z, sizeof(double) * (size_t)k);
        }
        for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) CAA[a * k + b] = C[(size_t)act[a] * n + act[b]];
        if (invert(CAA, inv, k)) {
          for (int a = 0; a < k; a++) {
            double t = 0;
            for (int b = 0; b < k; b++) t += inv[a * k + b] * u[act[b]];
            z[a] = t;
          }
          for (int i = 0; i < n; i++) {
            double t = 0;
            for (int a = 0; a < k; a++) t += C[(size_t)i * n + act[a]] * z[a];
            u[i] -= t;
          }
        }
        for (int a = 0; a < k; a++) u[act[a]] = 0;
      }
      double slope = 0;
      for (int i = 0; i < n; i++) slope -= gf[i] * u[i];
      edm = -0.5 * slope;
      if (edm < tol && edm >= 0) {
        converged = 1;
        break;
      }
      if (!(slope < 0)) {
        if (resets++ > 3) break;
        SET_DIAGONAL();
        continue;
      }

      // backtracking line search along -u, projected onto the bounds
      double t = 1, fn = INFINITY;
      int accepted = 0;
      for (int trial = 0; trial < 40; trial++) {
        memcpy(xn, x, sizeof(double) * (size_t)N);
        double predicted = 0;
        for (int i = 0; i < n; i++) {
          const int p = free_idx[i];
          const double v = clamp(x[p] - t * u[i], lower[p], upper[p]);
          xn[p] = v;
          if (v != x[p]) predicted += g[p] * (v - x[p]);
        }
        fn = hf_nll(m, data, constant, xn, gn);
        if (isfinite(fn) && fn <= f + 1e-4 * predicted) {
          accepted = 1;
          break;
        }
        const double quad = isfinite(fn) ? (-slope * t * t) / (2 * (fn - f - slope * t)) : 0;
        t = jmin(jmax(quad, 0.1 * t), 0.5 * t);
      }
      if (!accepted) {
        if (resets++ > 3) break;
        SET_DIAGONAL();
        continue;
      }

      double sy = 0, ss = 0, yy = 0;
      for (int i = 0; i < n; i++) {
        const int p = free_idx[i];
        s[i] = xn[p] - x[p];
        y[i] = gn[p] - g[p];
        sy += s[i] * y[i];
        ss += s[i] * s[i];
        yy += y[i] * y[i];
      }
      if (sy > 1e-10 * sqrt(ss * yy)) {
        double yCy = 0;
        for (int i = 0; i < n; i++) {
          double v = 0;
          const double *row = C + (size_t)i * n;
          for (int j = 0; j < n; j++) v += row[j] * y[j];
          Cy[i] = v;
          yCy += v * y[i];
        }
        const double a = (sy + yCy) / (sy * sy);
        for (int i = 0; i < n; i++) {
          double *row = C + (size_t)i * n;
          const double si = s[i], ci = Cy[i];
          for (int j = 0; j < n; j++) row[j] += a * si * s[j] - (ci * s[j] + si * Cy[j]) / sy;
        }
      }
      stalled = (f - fn < 1e-13 * (fabs(f) + 1)) ? stalled + 1 : 0;
      double *tmp = x; x = xn; xn = tmp;
      tmp = g; g = gn; gn = tmp;
      f = fn;
      if (stalled >= 3) break;
    }
  }
#undef SET_DIAGONAL
  memcpy(x_out, x, sizeof(double) * (size_t)N);
  memcpy(g_out, g, sizeof(double) * (size_t)N);
  *f_out = f;
  *edm_out = edm;
  *iter_out = iter;
  free(buf);
  free(act);
  free(CAA);
  free(inv);
  free(z);
  return converged;
}

// second derivatives of -2 ln L with respect to the k parameters in idx, from differences of the gradient
void hf_hessian(hf_model *m, const double *data, double constant, const double *x, const int32_t *idx, int k,
                const double *steps, int central, double *H) {
  const int N = m->n_pars;
  double *buf = (double *)calloc(4 * (size_t)N + 1, sizeof(double));
  double *xp = buf, *g0 = xp + N, *gp = g0 + N, *gm = gp + N;
  memcpy(xp, x, sizeof(double) * (size_t)N);
  if (!central) hf_nll(m, data, constant, xp, g0);
  for (int a = 0; a < k; a++) {
    const int p = idx[a];
    const double h = steps[a];
    xp[p] = x[p] + h;
    const int up = isfinite(hf_nll(m, data, constant, xp, gp));
    int down = 0;
    if (central || !up) {
      xp[p] = x[p] - h;
      down = isfinite(hf_nll(m, data, constant, xp, gm));
    }
    xp[p] = x[p];
    if ((!up || !down) && central) hf_nll(m, data, constant, xp, g0);
    double *row = H + (size_t)a * k;
    for (int b = 0; b < k; b++) {
      const int q = idx[b];
      if (up && down) row[b] = (gp[q] - gm[q]) / (2 * h);
      else if (up) row[b] = (gp[q] - g0[q]) / h;
      else if (down) row[b] = (g0[q] - gm[q]) / h;
      else row[b] = NAN;
    }
  }
  for (int a = 0; a < k; a++) {
    for (int b = a + 1; b < k; b++) {
      const double v = 0.5 * (H[(size_t)a * k + b] + H[(size_t)b * k + a]);
      H[(size_t)a * k + b] = H[(size_t)b * k + a] = v;
    }
  }
  free(buf);
}
