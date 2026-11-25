"""Statistical inference with a HistFactory model: fits, scans, ranking, limits, significance, yields.

Same procedures as fitter/js/infer.js; the minimiser and the second derivatives run in the C kernel.
"""

import copy
import ctypes
import math
from types import SimpleNamespace

import numpy as np

from . import _kernel

_lib = _kernel.lib


def norm_sf(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def _ratio(a, b):
    """a / b as floating-point division does it, without raising when b is zero"""
    if b != 0:
        return a / b
    return math.nan if a == 0 or math.isnan(a) else math.copysign(math.inf, a)


def linspace(a, b, n):
    return [a if n == 1 else a + ((b - a) * i) / (n - 1) for i in range(n)]


def interval(x, y, level=1):
    """minimum of y(x) and the points around it where y has risen by `level`"""
    i = min(range(len(y)), key=lambda k: y[k])
    y0 = y[i]

    def cross(a, b):
        return x[a] + ((level - (y[a] - y0)) / (y[b] - y[a])) * (x[b] - x[a])

    lo = hi = math.nan
    for k in range(i, 0, -1):
        if y[k - 1] - y0 >= level:
            lo = cross(k, k - 1)
            break
    for k in range(i, len(x) - 1):
        if y[k + 1] - y0 >= level:
            hi = cross(k, k + 1)
            break
    return SimpleNamespace(min=x[i], lo=lo, hi=hi, index=i)


def minimize(lik, x0, free, C=None, diag=None, tol=1e-7, max_iter=0):
    """Quasi-Newton minimisation of lik over the parameters in free, within the model's bounds."""
    model = lik.model
    n = len(free)
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    x = np.empty(model.n_pars)
    g = np.empty(model.n_pars)
    C_out = np.empty((n, n))
    active = np.zeros(n, dtype=np.uint8)
    f = ctypes.c_double()
    edm = ctypes.c_double()
    iterations = ctypes.c_int32()
    converged = _lib.hf_minimize(
        model._ref, lik._data, lik.constant, x0.ctypes.data, model.lower.ctypes.data, model.upper.ctypes.data,
        free.ctypes.data, n, None if C is None else C.ctypes.data, None if diag is None else diag.ctypes.data,
        tol, max_iter, x.ctypes.data, g.ctypes.data, C_out.ctypes.data, active.ctypes.data,
        ctypes.addressof(f), ctypes.addressof(edm), ctypes.addressof(iterations),
    )  # fmt: skip
    return SimpleNamespace(x=x, f=f.value, g=g, C=C_out, edm=edm.value, converged=bool(converged), iterations=iterations.value, active=active)


def hessian(lik, x, idx, steps, central):
    """second derivatives of lik with respect to the parameters in idx, from differences of the gradient"""
    k = len(idx)
    H = np.empty((k, k))
    _lib.hf_hessian(lik.model._ref, lik._data, lik.constant, x.ctypes.data, idx.ctypes.data, k, steps.ctypes.data, int(central), H.ctypes.data)
    return H


def _chol_inverse(H):
    """inverse of a symmetric matrix if it is positive definite, else None"""
    if H.size == 0:
        return H.copy()
    if not np.all(np.isfinite(H)):
        return None
    try:
        T = np.linalg.inv(np.linalg.cholesky(H))
    except np.linalg.LinAlgError:
        return None
    out = T.T @ T
    return 0.5 * (out + out.T)


def _inverse(H):
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None


def _diag_guess(model, idx):
    w = np.where(model.prefit_unc[idx] > 0, model.prefit_unc[idx], np.maximum((model.upper[idx] - model.lower[idx]) / 10, 1e-3))
    return 0.5 * w * w


def _step_size(model, x, idx):
    return 1e-4 * np.where(model.prefit_unc[idx] > 0, model.prefit_unc[idx], np.maximum(np.abs(x[idx]), 1))


def fit(model, lik, init=None, fixed=None):
    """Maximum likelihood fit with covariance. fixed: {index: value} to hold parameters constant."""
    N = model.n_pars
    x0 = np.array(model.init if init is None else init, dtype=np.float64)
    is_fixed = model.fixed.copy()
    for p, v in (fixed or {}).items():
        is_fixed[p] = 1
        x0[p] = v
    free = np.flatnonzero(is_fixed == 0).astype(np.int32)
    n = len(free)
    x0[free] = np.minimum(np.maximum(x0[free], model.lower[free]), model.upper[free])

    # start from the constraint widths: the second derivatives at the initial point are rarely positive definite
    # and cost more likelihood calls than they save
    res = minimize(lik, x0, free, diag=_diag_guess(model, free), tol=1e-6)
    x = res.x
    f = res.f

    # covariance from the second derivatives at the minimum, for parameters not stuck at a bound
    g = np.zeros(N)
    at_bound = np.zeros(N, dtype=np.uint8)
    inner = np.zeros(0, dtype=np.int32)
    cov = None
    pos_def = False
    edm = res.edm
    for _ in range(3 if n > 0 else 0):
        f = lik.value(x, g)
        stuck = ((x[free] <= model.lower[free]) & (g[free] > 0)) | ((x[free] >= model.upper[free]) & (g[free] < 0))
        at_bound[:] = 0
        at_bound[free[stuck]] = 1
        inner = free[~stuck]
        k = len(inner)
        H = hessian(lik, x, inner, _step_size(model, x, inner), True)
        Hinv = _chol_inverse(H)
        pos_def = Hinv is not None
        if not pos_def:
            Hinv = _inverse(H)
        if Hinv is None:
            break
        cov = 2 * Hinv
        if not pos_def:
            break
        # a few Newton steps with this matrix to land exactly on the minimum
        moved = 0.0
        lo, hi = model.lower[inner], model.upper[inner]
        sigma = np.sqrt(np.maximum(np.diag(cov), 1e-300))
        for _ in range(12):
            step = -(Hinv @ g[inner])
            edm = float(-0.5 * (g[inner] @ step))
            if edm < 1e-13:
                break
            trial = x.copy()
            gt = np.zeros(N)
            scale = 1.0
            ft = math.inf
            for _ in range(8):
                trial[inner] = np.minimum(np.maximum(x[inner] + scale * step, lo), hi)
                ft = lik.value(trial, gt)
                if ft <= f + 1e-12 * abs(f):
                    break
                scale *= 0.5
            if not ft <= f + 1e-12 * abs(f):
                break
            if k:
                moved = max(moved, float(np.max(np.abs(trial[inner] - x[inner]) / sigma)))
            x = trial
            f = ft
            g[:] = gt
        if moved < 0.02:
            break

    pos = np.full(N, -1, dtype=np.int32)
    pos[inner] = np.arange(len(inner))
    unc = np.zeros(N)
    if cov is not None:
        unc[inner] = np.sqrt(np.maximum(np.diag(cov), 0))
    return SimpleNamespace(
        pars=x,
        twice_nll=f,
        unc=unc,
        cov=cov,
        inner=inner,
        pos=pos,
        free=free,
        at_bound=at_bound,
        status=SimpleNamespace(converged=(res.converged or edm < 1e-4) and pos_def and math.isfinite(f), pos_def=pos_def, edm=edm, iterations=res.iterations),
        cache={},
    )


def cov_at(base, p, q):
    return 0.0 if base.pos[p] < 0 or base.pos[q] < 0 else base.cov[base.pos[p], base.pos[q]]


def conditional(model, lik, base, index, value, init=None):
    """Fit with parameters held at given values (one index and value, or sequences of them), starting
    from an earlier fit and its covariance. init: parameters to start from instead of the linear
    prediction from the covariance."""
    held = np.atleast_1d(np.asarray(index, dtype=np.int32))
    values = np.atleast_1d(np.asarray(value, dtype=np.float64))
    free = base.free[~np.isin(base.free, held)]
    n = len(free)
    cov = base.cov
    key = tuple(int(p) for p in held)
    start = base.cache.get(key)
    if start is None:
        # held parameters with an entry in the covariance matrix, and the inverse of their block of it
        known = [p for p in key if cov is not None and cov_at(base, p, p) > 0]
        at = base.pos[np.array(known, dtype=np.int32)]
        S_inv = _inverse(cov[np.ix_(at, at)]) if known else None
        if S_inv is None:
            known = []
        where = base.pos[free]
        inside = where >= 0 if cov is not None else np.zeros(n, dtype=bool)
        w = where[inside]
        rows = np.arange(n)
        # gain: shift of the free parameters in the covariance matrix per unit shift of the held ones; C: covariance of the free ones given the held ones
        gain = None
        C = np.zeros((n, n))
        C[rows[~inside], rows[~inside]] = _diag_guess(model, free[~inside])
        if cov is not None:
            sub = cov[np.ix_(w, w)]
            if known:
                gain = cov[np.ix_(w, at)] @ S_inv
                sub = sub - gain @ cov[np.ix_(at, w)]
            C[np.ix_(rows[inside], rows[inside])] = 0.5 * sub
        start = SimpleNamespace(C=C, gain=gain, known=known, moved=free[inside])
        base.cache[key] = start
    # linear prediction of where the other parameters go
    x0 = (base.pars if init is None else init).copy()
    x0[held] = values
    if init is None and start.known:
        shift = np.array([values[key.index(p)] - base.pars[p] for p in start.known])
        p = start.moved
        x0[p] = np.minimum(np.maximum(base.pars[p] + start.gain @ shift, model.lower[p]), model.upper[p])
    res = minimize(lik, x0, free, C=start.C, tol=1e-8)
    if not math.isfinite(res.f):
        # the prediction left the region where the model is defined: start from the base fit instead
        x1 = base.pars.copy()
        x1[held] = values
        res = minimize(lik, x1, free, C=start.C, tol=1e-8)
    return SimpleNamespace(pars=res.x, twice_nll=res.f, converged=res.converged)


# -2 Δln L at the edge of the 68.3% (one standard deviation) and 95% confidence regions, for one and for two parameters
LEVELS = {
    1: [SimpleNamespace(cl=0.6827, level=1.0), SimpleNamespace(cl=0.95, level=3.841458820694124)],
    2: [SimpleNamespace(cl=0.6827, level=2.295748928898636), SimpleNamespace(cl=0.95, level=5.991464547107979)],
}


def scan(model, lik, base, index, values, neighbours=False):
    """Profile likelihood scan of one parameter.

    intervals: for each confidence level (cl, level, ranges), where ranges lists the (from, to)
    stretches of the scan below the level (more than one if the likelihood has several minima), with
    the ends refined by further fits. An end is infinite if the curve is still below the level at the
    end of the scan.

    A fit can stop in a local minimum that the fit of a neighbouring point avoided. With neighbours,
    each point is fitted again from the solutions of its neighbours, going up and then down, and the
    lower result is kept: three fits per point instead of one. refits: number of points that ended
    lower than their first fit.
    """

    def lower(r, pars, value):
        """the fit started from pars if it ends below the result r, else r"""
        t = conditional(model, lik, base, index, value, init=pars)
        return t if t.twice_nll < r.twice_nll else r

    fits = [conditional(model, lik, base, index, v) for v in values]
    first = [r.twice_nll for r in fits]
    for step in (1, -1) if neighbours else ():
        for i in range(1, len(values)) if step == 1 else range(len(values) - 2, -1, -1):
            fits[i] = lower(fits[i], fits[i - step].pars, values[i])
    refits = sum(r.twice_nll < f - 1e-6 for r, f in zip(fits, first))
    out = [r.twice_nll - base.twice_nll for r in fits]
    iv = interval(values, out, 1)
    # measured from the fit result, or from the lowest point of the scan should that lie below it
    ref = min(out[iv.index], 0.0)
    last = len(values) - 1

    def root(y):
        return math.sqrt(max(y, 0.0))

    def crossing(level, k, step):
        """where the curve crosses the level between scan point k (below it) and its neighbour (above it)"""
        a = (values[k], out[k] - ref, fits[k].pars)
        b = (values[k + step], out[k + step] - ref, fits[k + step].pars)
        if not math.isfinite(b[1]):
            return b[0]
        # regula falsi on the square root, which is a straight line for a parabola
        x = b[0]
        for _ in range(8):
            x = a[0] + ((root(level) - root(a[1])) / (root(b[1]) - root(a[1]))) * (b[0] - a[0])
            # from the solutions at both ends, which may be different minima
            if neighbours:
                r = lower(conditional(model, lik, base, index, x, init=a[2]), b[2], x)
            else:
                r = conditional(model, lik, base, index, x)
            y = r.twice_nll - base.twice_nll - ref
            if not math.isfinite(y) or abs(y - level) < 1e-4:
                break
            if y < level:
                a = (x, y, r.pars)
            else:
                b = (x, y, r.pars)
        return x

    intervals = []
    for lv in LEVELS[1]:
        below = [y - ref < lv.level for y in out]
        ranges = []
        k = 0
        while k <= last:
            if below[k]:
                first = k
                while k < last and below[k + 1]:
                    k += 1
                ranges.append((-math.inf if first == 0 else crossing(lv.level, first, -1), math.inf if k == last else crossing(lv.level, k, 1)))
            k += 1
        intervals.append(SimpleNamespace(cl=lv.cl, level=lv.level, ranges=ranges))
    return SimpleNamespace(index=index, values=values, twice_delta_nll=out, interval=iv, reference=ref, intervals=intervals, refits=refits)


def scan2d(model, lik, base, indices, x, y):
    """Profile likelihood scan of two parameters on a grid: twice_delta_nll[i][j] for x[i], y[j]."""
    z = [[conditional(model, lik, base, indices, [u, v]).twice_nll - base.twice_nll for v in y] for u in x]
    return SimpleNamespace(indices=indices, x=x, y=y, twice_delta_nll=z, levels=LEVELS[2])


def approximate_impacts(model, base):
    """Impact of each nuisance parameter on the POI as estimated from the covariance matrix."""
    poi = model.poi_index
    out = []
    for p in base.inner:
        sigma = base.unc[p]
        if p == poi or not sigma > 0:
            continue
        c = cov_at(base, poi, p)
        out.append(SimpleNamespace(index=int(p), postfit=c / sigma, prefit=(c / (sigma * sigma)) * model.prefit_unc[p], exact=False))
    out.sort(key=lambda r: -abs(r.postfit))
    return out


def ranking(model, lik, base, indices):
    """Impacts from fits with the nuisance parameter held at its best-fit value +- uncertainty."""
    poi = model.poi_index
    mu = base.pars[poi]
    out = []
    for p in indices:

        def shift(d):
            if not d > 0:
                return [0.0, 0.0]
            return [conditional(model, lik, base, p, base.pars[p] + sign * d).pars[poi] - mu for sign in (1, -1)]

        post = shift(base.unc[p])
        pre = shift(model.prefit_unc[p])
        out.append(SimpleNamespace(index=p, postfit_up=post[0], postfit_down=post[1], prefit_up=pre[0], prefit_down=pre[1], exact=True))
    return out


def _asimov(model, lik, base, mu):
    """likelihood and starting point for the Asimov data set of the fit with the POI held at mu"""
    pars = conditional(model, lik, base, model.poi_index, mu).pars
    asimov_lik = model.likelihood(model.expected_data(pars))
    asimov_base = copy.copy(base)
    asimov_base.pars = pars
    asimov_base.twice_nll = asimov_lik.value(pars)
    asimov_base.cache = {}
    return asimov_lik, asimov_base


def tester(model, lik, base):
    """Hypothesis tests with the asymptotic formulae (arXiv:1007.1727).

    Returns a function mu -> cls, clsb, clb, expected[5] for (-2..+2 sigma), q, q_asimov.
    """
    poi = model.poi_index
    tilde = model.lower[poi] == 0
    mu_hat = base.pars[poi]
    asimov_lik, asimov_base = _asimov(model, lik, base, 0)

    def test(mu):
        q_obs = 0.0 if mu_hat > mu else max(conditional(model, lik, base, poi, mu).twice_nll - base.twice_nll, 0)
        q_a = max(conditional(model, asimov_lik, asimov_base, poi, mu).twice_nll - asimov_base.twice_nll, 0)
        root_q = math.sqrt(q_obs)
        root_a = math.sqrt(q_a)
        ts = root_q - root_a if not tilde or root_q <= root_a or root_a == 0 else (q_obs - q_a) / (2 * root_a)
        clsb = norm_sf(ts + root_a)
        clb = norm_sf(ts)
        return SimpleNamespace(
            mu=mu,
            cls=_ratio(clsb, clb),  # both underflow to zero far above the limit
            clsb=clsb,
            clb=clb,
            expected=[norm_sf(nsig + root_a) / norm_sf(nsig) for nsig in (2, 1, 0, -1, -2)],
            q=q_obs,
            q_asimov=q_a,
            tilde=tilde,
        )

    return test


def _crossing(x, y, level):
    """first place where y(x) drops below the level: (index of the point below, interpolated x)"""
    if not y[0] >= level:
        return -1, math.nan
    for i in range(1, len(x)):
        if y[i] < level:
            return i, x[i] + ((level - y[i]) / (y[i - 1] - y[i])) * (x[i - 1] - x[i])
    return -1, math.nan


def limit(model, lik, base, level=0.05, points=21, values=None):
    """Upper limit on the POI from a scan of CLs.

    With explicit values the limit is interpolated linearly between the scan points, as pyhf does.
    Otherwise the range is found automatically and each crossing is refined with further tests.
    """
    poi = model.poi_index
    test = tester(model, lik, base)
    explicit = values is not None
    if not explicit:
        # find a range in which all curves drop below the level
        top = model.upper[poi]
        sigma = base.unc[poi] if base.unc[poi] > 0 else top / 20
        hi = min(top, max(base.pars[poi], 0) + 3 * sigma)
        for _ in range(12):
            r = test(hi)
            worst = max(r.cls, r.expected[4])
            if worst > level / 5 and hi < top:
                hi = min(top, hi * 1.7)
            elif worst < level / 500:
                hi *= 0.6
            else:
                break
        values = linspace(0, hi, points)
    results = [test(v) for v in values]
    curves = [lambda r: r.cls] + [lambda r, c=c: r.expected[c] for c in range(5)]
    limits = []
    for get in curves:
        i, x = _crossing([r.mu for r in results], [get(r) for r in results], level)
        if not explicit and i > 0:
            # regula falsi on ln(CLs) between the two scan points around the crossing
            a = results[i - 1]
            b = results[i]
            for _ in range(8):
                fa = math.log(get(a) / level)
                fb = math.log(max(get(b), 1e-300) / level)
                x = a.mu - (fa * (b.mu - a.mu)) / (fb - fa)
                if abs(b.mu - a.mu) < 1e-4 * x:
                    break
                r = test(x)
                results.append(r)
                if abs(get(r) / level - 1) < 1e-4:
                    break
                if get(r) >= level:
                    a = r
                else:
                    b = r
        limits.append(x)
    results.sort(key=lambda r: r.mu)
    return SimpleNamespace(level=level, values=[r.mu for r in results], results=results, observed=limits[0], expected=limits[1:], tilde=results[0].tilde)


def significance(model, lik, base):
    """Discovery test statistic q0 with observed and expected significance."""
    poi = model.poi_index
    bkg_only = conditional(model, lik, base, poi, 0)
    q0 = max(bkg_only.twice_nll - base.twice_nll, 0) if base.pars[poi] > 0 else 0.0
    asimov_lik, asimov_base = _asimov(model, lik, base, 1)
    q0_a = max(conditional(model, asimov_lik, asimov_base, poi, 0).twice_nll - asimov_base.twice_nll, 0)
    return SimpleNamespace(q0=q0, z=math.sqrt(q0), p0=norm_sf(math.sqrt(q0)), z_expected=math.sqrt(q0_a), p0_expected=norm_sf(math.sqrt(q0_a)))


def yields(model, pars, base=None):
    """Yields per channel, sample and bin with uncertainties from linear error propagation.

    Without a fit result the pre-fit uncertainties of the constraint terms are used. The uncertainties
    are NaN if the fit has no covariance matrix.
    """
    pars = np.ascontiguousarray(pars, dtype=np.float64)
    grad = np.zeros(model.n_pars)
    floating = model.fixed == 0
    model.evaluate(pars)
    cell_nu = model.cell_nu.copy()
    w = model.cell_weight

    def uncertainty(blocks, lo, hi):
        # uncertainty of the sum over the given blocks, restricted to bins lo..hi of the channel
        w[:] = 0
        for blk in blocks:
            w[blk.cell + lo : blk.cell + hi] = 1
        grad[:] = 0
        model.accumulate(pars, grad, np.array([blk.id for blk in blocks], dtype=np.int32))
        if base is None:
            return math.sqrt(float(np.sum(np.square(grad[floating] * model.prefit_unc[floating]))))
        if base.cov is None:
            return math.nan
        gi = grad[base.inner]
        return math.sqrt(max(float(gi @ base.cov @ gi), 0))

    channels = []
    for ci, ch in enumerate(model.channels):
        blocks = model.channel_blocks[ci]
        total = np.zeros(ch.n_bins)
        samples = []
        for blk in blocks:
            y = cell_nu[blk.cell : blk.cell + ch.n_bins]
            total += y
            samples.append(SimpleNamespace(name=model.samples[blk.sample], yields=y, sum=float(y.sum()), sum_unc=uncertainty([blk], 0, ch.n_bins)))
        data = model.observed[ch.offset : ch.offset + ch.n_bins]
        channels.append(
            SimpleNamespace(
                name=ch.name,
                data=data,
                data_sum=float(data.sum()),
                total=total,
                total_unc=[uncertainty(blocks, b, b + 1) for b in range(ch.n_bins)],
                samples=samples,
                sum=float(total.sum()),
                sum_unc=uncertainty(blocks, 0, ch.n_bins),
            )
        )
    w[:] = 0
    return SimpleNamespace(channels=channels)


def correlation(base):
    sigma = np.sqrt(np.diag(base.cov))
    return base.cov / np.outer(sigma, sigma)
