"""Reference numbers from pyhf for the JavaScript fitter.

Writes the test workspaces to test/fitter/workspaces/ and the numbers to test/fitter/reference.json.
Run with: pixi run python test/fitter/export_reference.py [workspace names]
"""

import json
import pathlib
import time

import numpy as np
import pyhf

HERE = pathlib.Path(__file__).parent
(HERE / "workspaces").mkdir(exist_ok=True)


def workspace(channels, observations, poi="mu", parameters=None):
    return {
        "channels": channels,
        "observations": [{"name": k, "data": v} for k, v in observations.items()],
        "measurements": [{"name": "meas", "config": {"poi": poi, "parameters": parameters or []}}],
        "version": "1.0.0",
    }


def mod(name, type_, data=None):
    return {"name": name, "type": type_, "data": data}


def ws_shapesys():
    return workspace(
        [
            {
                "name": "SR",
                "samples": [
                    {"name": "signal", "data": [12.0, 11.0], "modifiers": [mod("mu", "normfactor")]},
                    {"name": "background", "data": [50.0, 52.0], "modifiers": [mod("uncorr_bkguncrt", "shapesys", [3.0, 7.0])]},
                ],
            }
        ],
        {"SR": [51.0, 48.0]},
    )


def ws_histosys():
    return workspace(
        [
            {
                "name": "SR",
                "samples": [
                    {"name": "signal", "data": [5.0, 10.0, 6.0], "modifiers": [mod("mu", "normfactor")]},
                    {
                        "name": "background",
                        "data": [50.0, 60.0, 40.0],
                        "modifiers": [
                            mod("shape", "histosys", {"hi_data": [55.0, 60.0, 37.0], "lo_data": [46.0, 61.0, 44.0]}),
                            mod("norm", "normsys", {"hi": 1.1, "lo": 0.92}),
                        ],
                    },
                ],
            }
        ],
        {"SR": [62.0, 71.0, 41.0]},
    )


def ws_everything():
    """all modifier types, shared names, a sample missing in one channel, zero uncertainties, parameter configs"""
    return workspace(
        [
            {
                "name": "SR",
                "samples": [
                    {
                        "name": "signal",
                        "data": [2.0, 6.0, 9.0],
                        "modifiers": [
                            mod("mu", "normfactor"),
                            mod("lumi", "lumi"),
                            mod("acc", "normsys", {"hi": 1.08, "lo": 0.9}),
                            mod("jes", "histosys", {"hi_data": [2.2, 6.3, 9.1], "lo_data": [1.9, 5.6, 8.8]}),
                            mod("jes", "normsys", {"hi": 1.03, "lo": 0.98}),
                            mod("stat_SR", "staterror", [0.2, 0.3, 0.4]),
                        ],
                    },
                    {
                        "name": "ttbar",
                        "data": [40.0, 25.0, 10.0],
                        "modifiers": [
                            mod("k_tt", "normfactor"),
                            mod("lumi", "lumi"),
                            mod("jes", "histosys", {"hi_data": [44.0, 26.0, 9.0], "lo_data": [37.0, 24.5, 10.5]}),
                            mod("stat_SR", "staterror", [2.0, 1.5, 1.0]),
                            mod("gen", "normsys", {"hi": 1.25, "lo": 0.7}),
                        ],
                    },
                    {
                        "name": "fakes",
                        "data": [8.0, 4.0, 0.0],
                        "modifiers": [mod("fake_bins", "shapesys", [2.0, 0.0, 0.0])],
                    },
                ],
            },
            {
                "name": "CR",
                "samples": [
                    {
                        "name": "ttbar",
                        "data": [400.0, 300.0],
                        "modifiers": [
                            mod("k_tt", "normfactor"),
                            mod("lumi", "lumi"),
                            mod("jes", "histosys", {"hi_data": [420.0, 330.0], "lo_data": [385.0, 280.0]}),
                            mod("stat_CR", "staterror", [8.0, 0.0]),
                            mod("gen", "normsys", {"hi": 1.25, "lo": 0.7}),
                        ],
                    },
                    {
                        "name": "multijet",
                        "data": [50.0, 20.0],
                        "modifiers": [mod("qcd_shape", "shapefactor"), mod("stat_CR", "staterror", [5.0, 0.0])],
                    },
                    {
                        "name": "fakes",
                        "data": [30.0, 12.0],
                        "modifiers": [mod("fake_norm", "normsys", {"hi": 1.5, "lo": 0.5}), mod("frozen", "normsys", {"hi": 1.2, "lo": 0.8})],
                    },
                ],
            },
        ],
        {"SR": [55.0, 38.0, 22.0], "CR": [470.0, 345.0]},
        parameters=[
            {"name": "lumi", "inits": [1.0], "bounds": [[0.5, 1.5]], "auxdata": [1.0], "sigmas": [0.02]},
            {"name": "mu", "inits": [1.0], "bounds": [[0.0, 8.0]]},
            {"name": "frozen", "fixed": True, "inits": [0.5]},
            {"name": "gen", "bounds": [[-3.0, 3.0]]},
        ],
    )


def ws_random(seed, n_channels, n_samples, n_bins, n_norm, n_shape, staterror=True):
    """workspace with many correlated systematics; data fluctuated around a shifted model"""
    rng = np.random.default_rng(seed)
    channels = []
    observations = {}
    samples = [f"bkg{k}" for k in range(n_samples - 1)]
    norm_names = [f"norm_{k:03d}" for k in range(n_norm)]
    shape_names = [f"shape_{k:03d}" for k in range(n_shape)]
    for c in range(n_channels):
        nb = int(n_bins if np.isscalar(n_bins) else n_bins[c])
        x = np.linspace(0, 1, nb + 1)[:-1]
        cname = f"region_{c:02d}"
        signal_fraction = 0.02 + 0.2 * c / max(n_channels - 1, 1)
        channel_samples = []
        total = np.zeros(nb)
        for s, sname in enumerate(samples):
            if rng.random() < 0.15 and s > 0:
                continue
            scale = rng.uniform(20, 400) / (1 + s)
            nominal = scale * np.exp(-rng.uniform(0.5, 3) * x) + 1.0
            mods = []
            if s < 2:
                mods.append(mod(f"k_{sname}", "normfactor"))
            for name in norm_names:
                if rng.random() < 0.3:
                    size = rng.uniform(0.01, 0.15)
                    mods.append(mod(name, "normsys", {"hi": 1 + size, "lo": 1 - size * rng.uniform(0.7, 1.2)}))
            for name in shape_names:
                if rng.random() < 0.3:
                    tilt = rng.uniform(-0.1, 0.1) * (x - 0.5) + rng.uniform(-0.03, 0.03)
                    hi = nominal * (1 + tilt)
                    lo = nominal * (1 - tilt * rng.uniform(0.7, 1.2))
                    mods.append(mod(name, "histosys", {"hi_data": hi.tolist(), "lo_data": lo.tolist()}))
            if staterror:
                mods.append(mod(f"staterror_{cname}", "staterror", (nominal * rng.uniform(0.01, 0.06, nb)).tolist()))
            channel_samples.append({"name": sname, "data": nominal.tolist(), "modifiers": mods})
            total += nominal
        signal = signal_fraction * total.sum() * np.exp(-0.5 * ((x - 0.7) / 0.15) ** 2)
        signal = signal / signal.sum() * signal_fraction * total.sum() + 0.01
        mods = [mod("mu", "normfactor"), mod("norm_000", "normsys", {"hi": 1.05, "lo": 0.95})]
        if staterror:
            mods.append(mod(f"staterror_{cname}", "staterror", (signal * 0.03).tolist()))
        channel_samples.append({"name": "signal", "data": signal.tolist(), "modifiers": mods})
        channels.append({"name": cname, "samples": channel_samples})
        observations[cname] = None

    spec = workspace(channels, {k: [0.0] * len(c["samples"][0]["data"]) for k, c in zip(observations, channels)})
    model = pyhf.Workspace(spec, validate=False).model(validate=False)
    truth = np.array(model.config.suggested_init())
    for i, name in enumerate(model.config.par_names):
        if name.startswith(("norm_", "shape_")):
            truth[i] = rng.normal(0, 0.7)
        elif name.startswith("k_"):
            truth[i] = rng.uniform(0.85, 1.15)
        elif name == "mu":
            truth[i] = 0.6
    expected = np.asarray(model.expected_actualdata(truth))
    data = rng.poisson(expected).astype(float)
    start = 0
    for obs, c in zip(spec["observations"], model.config.channels):
        nb = model.config.channel_nbins[c]
        obs["name"] = c
        obs["data"] = data[start : start + nb].tolist()
        start += nb
    return spec


WORKSPACES = {
    "shapesys": (ws_shapesys(), "full"),
    "histosys": (ws_histosys(), "full"),
    "everything": (ws_everything(), "full"),
    "medium": (ws_random(1, 3, 4, 6, 8, 6), "full"),
    "large": (ws_random(2, 10, 8, [12, 8, 15, 10, 6, 20, 9, 14, 10, 12], 120, 80), "fit"),
}


def random_points(model, rng, n):
    init = np.array(model.config.suggested_init())
    bounds = np.array(model.config.suggested_bounds())
    fixed = np.array(model.config.suggested_fixed())
    points = []
    for k in range(n):
        width = [0.3, 1.0, 2.5][k % 3]
        p = init.copy()
        for i, name in enumerate(model.config.par_names):
            constrained_at_zero = init[i] == 0
            step = rng.normal(0, width) if constrained_at_zero else rng.normal(0, 0.1 * width)
            if not fixed[i]:
                p[i] = np.clip(init[i] + step, bounds[i][0] + 1e-3, bounds[i][1] - 1e-3)
        points.append(p.tolist())
    return points


def export(name, spec, level):
    print(f"{name}: ", end="", flush=True)
    t0 = time.time()
    (HERE / "workspaces" / f"{name}.json").write_text(json.dumps(spec))
    ws = pyhf.Workspace(spec, validate=False)
    model = ws.model(validate=False)
    data = np.asarray(ws.data(model))
    rng = np.random.default_rng(42)
    out = {
        "par_names": model.config.par_names,
        "channels": model.config.channels,
        "samples": model.config.samples,
        "init": model.config.suggested_init(),
        "bounds": [list(b) for b in model.config.suggested_bounds()],
        "fixed": [bool(f) for f in model.config.suggested_fixed()],
        "auxdata": list(model.config.auxdata),
        "data": data.tolist(),
        "poi_index": model.config.poi_index,
        "points": [],
    }
    for p in random_points(model, rng, 6):
        out["points"].append(
            {
                "pars": p,
                "expected_data": np.asarray(model.expected_data(p)).tolist(),
                "logpdf": float(model.logpdf(p, data)[0]),
            }
        )

    init = model.config.suggested_init()
    bounds = model.config.suggested_bounds()
    fixed = model.config.suggested_fixed()
    poi = model.config.poi_index

    # maximum likelihood fit with uncertainties and correlations
    if level != "full":
        # many parameters: Minuit with numerical gradients does not finish within millions of calls,
        # so only the minimum found by scipy is recorded (a looser reference than Minuit's)
        pyhf.set_backend("numpy", pyhf.optimize.scipy_optimizer(maxiter=200_000))
        _, twice_nll = pyhf.infer.mle.fit(data, model, return_fitted_val=True)
        out["scipy_twice_nll"] = float(twice_nll)
        print(f"scipy fit {time.time() - t0:.1f}s")
        return out
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(tolerance=1e-3, strategy=2))
    result, twice_nll, opt = pyhf.infer.mle.fit(
        data, model, return_uncertainties=True, return_fitted_val=True, return_result_obj=True
    )
    out["fit"] = {
        "bestfit": result[:, 0].tolist(),
        "uncertainty": result[:, 1].tolist(),
        "twice_nll": float(twice_nll),
        "correlation": np.asarray(opt.corr).tolist() if level == "full" else None,
    }
    if level == "full":
        # Minuit's uncertainties are distorted for parameters with a range comparable to their uncertainty,
        # so the reference covariance is the inverse of the numerical second derivatives of pyhf's likelihood
        best = result[:, 0]
        # parameters that Minuit leaves next to a bound are treated as sitting on it
        def inside(i):
            margin = 1e-3 * (bounds[i][1] - bounds[i][0])
            return bounds[i][0] + margin < best[i] < bounds[i][1] - margin

        free = [i for i in range(len(best)) if not fixed[i] and inside(i)]
        step = 1e-3

        def nll(shifts):
            x = best.copy()
            for i, d in shifts:
                x[i] += d
            return -2 * float(model.logpdf(x, data)[0])

        f0 = nll([])
        k = len(free)
        hess = np.zeros((k, k))
        plus = [nll([(i, step)]) for i in free]
        minus = [nll([(i, -step)]) for i in free]
        for a in range(k):
            hess[a, a] = (plus[a] - 2 * f0 + minus[a]) / step**2
            for b in range(a):
                pp = nll([(free[a], step), (free[b], step)])
                mm = nll([(free[a], -step), (free[b], -step)])
                hess[a, b] = hess[b, a] = (pp - plus[a] - plus[b] + 2 * f0 - minus[a] - minus[b] + mm) / (2 * step**2)
        cov = 2 * np.linalg.inv(hess)
        sigma = np.sqrt(np.diag(cov))
        out["hesse"] = {"free": free, "uncertainty": sigma.tolist(), "correlation": (cov / np.outer(sigma, sigma)).tolist()}
    print(f"fit {time.time() - t0:.1f}s ", end="", flush=True)

    # conditional fits and hypothesis tests with a looser tolerance (strict settings can fail on Asimov data)
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(tolerance=1e-2))
    out["fixed_poi"] = []
    for mu in [0.0, 0.5, 2.0]:
        _, nll = pyhf.infer.mle.fixed_poi_fit(mu, data, model, return_fitted_val=True)
        out["fixed_poi"].append({"mu": mu, "twice_nll": float(nll)})

    if level == "full":
        # hypothesis tests
        out["hypotest"] = []
        for mu in [0.3, 1.0, 2.5]:
            cls_obs, tails, cls_exp = pyhf.infer.hypotest(
                mu, data, model, test_stat="qtilde", return_tail_probs=True, return_expected_set=True
            )
            qmu_tilde = pyhf.infer.test_statistics.qmu_tilde(mu, data, model, init, bounds, fixed)
            out["hypotest"].append(
                {
                    "mu": mu,
                    "cls": float(cls_obs),
                    "clsb": float(tails[0]),
                    "clb": float(tails[1]),
                    "expected": [float(v) for v in cls_exp],
                    "qmu_tilde": float(qmu_tilde),
                }
            )
        # the same with negative signal strengths allowed (q_mu)
        wide = [list(b) for b in bounds]
        wide[poi] = [-5.0, bounds[poi][1]]
        out["hypotest_q"] = []
        for mu in [1.0]:
            try:
                cls_obs, cls_exp = pyhf.infer.hypotest(
                    mu, data, model, par_bounds=wide, test_stat="q", return_expected_set=True
                )
            except pyhf.exceptions.FailedMinimization:
                print("(q_mu test failed in pyhf) ", end="")
                continue
            out["hypotest_q"].append({"mu": mu, "cls": float(cls_obs), "expected": [float(v) for v in cls_exp]})
        print(f"tests {time.time() - t0:.1f}s ", end="", flush=True)

        scan = np.linspace(0.05, bounds[poi][1] * 0.6, 25)
        obs_limit, exp_limits, (_, results) = pyhf.infer.intervals.upper_limits.upper_limit(
            data, model, scan, return_results=True
        )
        out["limit"] = {
            "scan": scan.tolist(),
            "observed": float(obs_limit),
            "expected": [float(v) for v in exp_limits],
        }

        # limits from root finding rather than from interpolation on a coarse scan
        try:
            obs_fine, exp_fine = pyhf.infer.intervals.upper_limits.toms748_scan(
                data, model, 0.0, bounds[poi][1], rtol=1e-5
            )
            out["limit_fine"] = {"observed": float(obs_fine), "expected": [float(v) for v in exp_fine]}
        except Exception as e:  # noqa: BLE001
            print(f"(root finding failed: {e}) ", end="")

        # discovery
        p0, p0_exp = pyhf.infer.hypotest(0.0, data, model, test_stat="q0", return_expected=True)
        q0 = pyhf.infer.test_statistics.q0(0.0, data, model, init, bounds, fixed)
        out["discovery"] = {"q0": float(q0), "p0": float(p0), "p0_expected": float(p0_exp)}

    print(f"done {time.time() - t0:.1f}s")
    return out


if __name__ == "__main__":
    import sys

    # without arguments everything is exported; with names only those entries are replaced
    path = HERE / "reference.json"
    reference = json.loads(path.read_text()) if path.exists() and sys.argv[1:] else {}
    for name in sys.argv[1:] or WORKSPACES:
        spec, level = WORKSPACES[name]
        reference[name] = export(name, spec, level)
        path.write_text(json.dumps(reference))
