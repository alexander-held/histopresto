"""HistFactory model built from a pyhf workspace, flattened into arrays for the C kernel.

Channel, sample and parameter ordering follow pyhf, so vectors compare element by element.
"""

import math
from types import SimpleNamespace

import numpy as np

from . import _kernel
from .expression import compile_bindings, parse_bindings
from .workspace import MODIFIER_TYPES

_lib = _kernel.lib
_MISSING = object()

A_INVERSE = [
    [15 / 16, -15 / 16, -7 / 16, -7 / 16, 1 / 16, -1 / 16],
    [3 / 2, 3 / 2, -9 / 16, 9 / 16, 1 / 16, 1 / 16],
    [-5 / 8, 5 / 8, 5 / 8, 5 / 8, -1 / 8, 1 / 8],
    [-3 / 2, -3 / 2, 7 / 8, -7 / 8, -1 / 8, -1 / 8],
    [3 / 16, -3 / 16, -3 / 16, -3 / 16, 1 / 16, -1 / 16],
    [1 / 2, 1 / 2, -5 / 16, 5 / 16, 1 / 16, 1 / 16],
]


def code4_coefficients(hi, lo):
    """polynomial coefficients of interpolation code 4 for |alpha| < 1"""
    lu = math.log(hi)
    ld = math.log(lo)
    b = [hi - 1, lo - 1, lu * hi, -ld * lo, lu * lu * hi, ld * ld * lo]
    out = []
    for row in A_INVERSE:
        s = 0.0
        for a, v in zip(row, b):
            s += a * v
        out.append(s)
    return out


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _scalar(kind, init, bounds, **extra):
    return {"kind": kind, "n": 1, "init": [init], "bounds": [bounds], "fixed": [False], **extra}


class Likelihood:
    """-2 ln L for a data vector (main data followed by auxiliary data)"""

    def __init__(self, model, data):
        self.model = model
        self.data = _f64(data)
        self._data = self.data.ctypes.data
        self.constant = _lib.hf_constant(model._ref, self._data)

    def value(self, pars, grad=None):
        pars = _f64(pars)
        return _lib.hf_nll(self.model._ref, self._data, self.constant, pars.ctypes.data, None if grad is None else grad.ctypes.data)


class Model:
    def __init__(self, ws, measurement_index=0, poi=None, poi_bounds=None):
        measurement = ws["measurements"][measurement_index]
        self.measurement = measurement["name"]

        # ----- channels, samples, modifiers (all sorted, as in pyhf) -----
        channel_specs = {c["name"]: c for c in ws["channels"]}
        modifier_keys = {}
        sample_set = set()
        for c in ws["channels"]:
            for s in c["samples"]:
                sample_set.add(s["name"])
                for m in s["modifiers"]:
                    modifier_keys[f"{m['name']}\0{m['type']}"] = (m["name"], m["type"])
        self.samples = sorted(sample_set)
        modifiers = [modifier_keys[k] for k in sorted(modifier_keys)]

        # expressions of purefunc modifiers; their symbols are parameters
        trees, symbols = parse_bindings(ws.get("bindings", []))
        for name, type_ in modifiers:
            if type_ == "purefunc" and name not in trees:
                raise ValueError(f'The purefunc modifier "{name}" refers to a binding that is not declared.')

        self.channels = []
        offset = 0
        for name in sorted(channel_specs):
            n_bins = len(channel_specs[name]["samples"][0]["data"])
            self.channels.append(SimpleNamespace(name=name, n_bins=n_bins, offset=offset))
            offset += n_bins
        self.n_bins = B = offset

        # defined (channel, sample) combinations with their modifiers by type
        defined = []
        for ch in self.channels:
            by_sample = {s["name"]: (s, {f"{m['type']}/{m['name']}": m for m in s["modifiers"]}) for s in channel_specs[ch.name]["samples"]}
            defined.append([by_sample.get(sn) for sn in self.samples])

        # ----- parameter sets -----
        requirements = {}  # name -> list of requirements
        stat = {}  # staterror name -> global bin indices
        shape_blocks = {}  # shapesys / shapefactor name -> channel index

        for type_ in MODIFIER_TYPES:
            if type_ == "staterror":
                for name, t in modifiers:
                    if t != type_:
                        continue
                    key = f"{type_}/{name}"
                    nom_sum = np.zeros(B)
                    unc2 = np.zeros(B)
                    mask = np.zeros(B, dtype=bool)
                    for si in range(len(self.samples)):
                        if not any(defined[ci][si] and key in defined[ci][si][1] for ci in range(len(self.channels))):
                            continue
                        for ci, ch in enumerate(self.channels):
                            d = defined[ci][si]
                            if not d:
                                continue
                            sl = slice(ch.offset, ch.offset + ch.n_bins)
                            nom_sum[sl] += d[0]["data"]
                            mod = d[1].get(key)
                            if mod:
                                unc2[sl] += np.square(_f64(mod["data"]))
                                mask[sl] = True
                    bins = [int(b) for b in np.flatnonzero(mask)]
                    sigmas = [math.sqrt(unc2[b]) / nom_sum[b] if nom_sum[b] > 0 else 0.0 for b in bins]
                    stat[name] = bins
                    requirements.setdefault(name, []).append(
                        {
                            "kind": "normal",
                            "n": len(bins),
                            "init": [1] * len(bins),
                            "bounds": [[1e-10, 10]] * len(bins),
                            "fixed": [s == 0 for s in sigmas],
                            "aux": [1] * len(bins),
                            "sigmas": [1 if s == 0 else s for s in sigmas],
                        }
                    )
                continue
            if type_ == "purefunc":
                for name in symbols:
                    requirements.setdefault(name, []).append(_scalar("free", 1, [0, 10]))
                continue
            seen = set()
            for ci in range(len(self.channels)):
                for si in range(len(self.samples)):
                    d = defined[ci][si]
                    if not d:
                        continue
                    for name, t in modifiers:
                        if t != type_:
                            continue
                        mod = d[1].get(f"{type_}/{name}")
                        if not mod:
                            continue
                        if type_ in ("shapesys", "shapefactor"):
                            if name in shape_blocks and (type_ == "shapesys" or shape_blocks[name] != ci):
                                raise ValueError(
                                    f'The shapesys modifier "{name}" is used more than once; shapesys modifiers cannot be shared between samples or channels.'
                                    if type_ == "shapesys"
                                    else f'The shapefactor modifier "{name}" is used in more than one channel, which is not supported.'
                                )
                            shape_blocks[name] = ci
                        if name in seen:
                            continue
                        seen.add(name)
                        nominal = d[0]["data"]
                        if type_ in ("histosys", "normsys"):
                            req = _scalar("normal", 0, [-5, 5], aux=[0])
                        elif type_ == "normfactor":
                            req = _scalar("free", 1, [0, 10])
                        elif type_ == "lumi":
                            req = _scalar("normal", None, None, aux=None, sigmas=None)
                        elif type_ == "shapefactor":
                            n = len(nominal)
                            req = {"kind": "free", "n": n, "init": [1] * n, "bounds": [[0, 10]] * n, "fixed": [False] * n}
                        else:  # shapesys
                            valid = [nom > 0 and unc > 0 for nom, unc in zip(nominal, mod["data"])]
                            factors = [(nom * nom) / (unc * unc) if v else 1 for nom, unc, v in zip(nominal, mod["data"], valid)]
                            n = len(factors)
                            req = {
                                "kind": "poisson",
                                "n": n,
                                "init": [1] * n,
                                "bounds": [[1e-10, 10]] * n,
                                "fixed": [not v for v in valid],
                                "aux": factors,
                                "factors": factors,
                            }
                        requirements.setdefault(name, []).append(req)
        if not requirements:
            raise ValueError("The model has no parameters.")

        user_config = {}
        for p in measurement["config"]["parameters"]:
            if p["name"] in user_config:
                raise ValueError(f'Measurement "{measurement["name"]}" configures parameter "{p["name"]}" more than once.')
            user_config[p["name"]] = p

        self.sets = []
        set_by_name = {}
        n_pars = 0
        for name, reqs in requirements.items():
            s = dict(reqs[0])
            for other in reqs[1:]:
                for k in ("kind", "n", "init", "bounds", "fixed", "aux", "sigmas", "factors"):
                    if s.get(k, _MISSING) != other.get(k, _MISSING):
                        raise ValueError(f'The name "{name}" is used by modifiers of incompatible types; use unique modifier names.')
            user = user_config.get(name, {})
            for key, user_key in (("init", "inits"), ("bounds", "bounds"), ("aux", "auxdata"), ("sigmas", "sigmas"), ("factors", "factors")):
                if user_key not in user:
                    continue
                v = user[user_key]
                if key not in s:
                    raise ValueError(f'Parameter "{name}" does not use the "{user_key}" setting.')
                if not isinstance(v, list) or len(v) != s["n"]:
                    raise ValueError(f'Parameter "{name}": "{user_key}" needs {s["n"]} value{"" if s["n"] == 1 else "s"}.')
                s[key] = v
            if isinstance(user.get("fixed"), bool):
                s["fixed"] = [user["fixed"]] * s["n"]
            if s["init"][0] is None or s["bounds"][0] is None or s.get("aux", _MISSING) is None or s.get("sigmas", _MISSING) is None:
                raise ValueError(
                    f'The luminosity parameter "{name}" needs inits, bounds, auxdata and sigmas in the measurement\'s parameter configuration.'
                )
            s.update(name=name, start=n_pars, types=[])
            n_pars += s["n"]
            self.sets.append(s)
            set_by_name[name] = s
        for name, type_ in modifiers:
            if type_ != "purefunc":
                set_by_name[name]["types"].append(type_)
        for name in symbols:
            set_by_name[name]["types"].append("purefunc")
        self.n_pars = n_pars

        # flat per-parameter arrays
        self.names = []
        self.set_index = np.zeros(n_pars, dtype=np.int32)
        self.init = np.zeros(n_pars)
        self.lower = np.zeros(n_pars)
        self.upper = np.zeros(n_pars)
        self.fixed = np.zeros(n_pars, dtype=np.uint8)
        self.prefit_unc = np.zeros(n_pars)
        self.aux_index = np.full(n_pars, -1, dtype=np.int32)
        gauss_idx, gauss_sigma, pois_idx, pois_tau = [], [], [], []
        auxdata = []
        for k, s in enumerate(self.sets):
            plain = s["n"] == 1 and all(t not in ("staterror", "shapesys", "shapefactor") for t in s["types"])
            for i in range(s["n"]):
                p = s["start"] + i
                self.names.append(s["name"] if plain else f"{s['name']}[{i}]")
                self.set_index[p] = k
                self.init[p] = s["init"][i]
                self.lower[p], self.upper[p] = s["bounds"][i]
                self.fixed[p] = 1 if s["fixed"][i] else 0
                if s["kind"] == "free":
                    continue
                self.aux_index[p] = len(auxdata)
                auxdata.append(s["aux"][i])
                if s["kind"] == "normal":
                    sigma = s["sigmas"][i] if s.get("sigmas") else 1
                    gauss_idx.append(p)
                    gauss_sigma.append(sigma)
                    self.prefit_unc[p] = sigma
                else:
                    pois_idx.append(p)
                    pois_tau.append(s["factors"][i])
                    self.prefit_unc[p] = 1 / math.sqrt(s["factors"][i])
        self.auxdata = _f64(auxdata)
        self.n_aux = len(auxdata)
        self.gauss_idx = np.array(gauss_idx, dtype=np.int32)
        self.gauss_sigma = _f64(gauss_sigma)
        self.pois_idx = np.array(pois_idx, dtype=np.int32)
        self.pois_tau = _f64(pois_tau)

        # parameter of interest
        poi = poi if poi is not None else measurement["config"]["poi"]
        self.poi_index = -1
        self.poi_name = ""
        if poi:
            s = set_by_name.get(poi)
            if not s:
                raise ValueError(f'The parameter of interest "{poi}" does not appear in the model.')
            if s["n"] != 1:
                raise ValueError(f'The parameter of interest "{poi}" has several components, which is not supported.')
            p = self.poi_index = s["start"]
            self.poi_name = poi
            if poi_bounds:
                self.lower[p], self.upper[p] = poi_bounds
                self.init[p] = min(max(self.init[p], self.lower[p]), self.upper[p])

        # observed data in model order, followed by the auxiliary data
        self.observed = np.zeros(B + self.n_aux)
        for ch in self.channels:
            self.observed[ch.offset : ch.offset + ch.n_bins] = ws["observations"][ch.name]
        self.observed[B:] = self.auxdata

        # ----- blocks, one per defined (channel, sample), as flat tables with offsets -----
        stat_index = {}
        for name, bins in stat.items():
            index = np.full(B, -1, dtype=np.int32)
            index[bins] = set_by_name[name]["start"] + np.arange(len(bins))
            stat_index[name] = index
        t = {k: [] for k in ("blk_offset", "blk_n", "blk_cell", "nominal", "scal_start", "scal_count", "scal_par", "ns_start", "ns_count", "ns_par",
                             "ns_lnhi", "ns_lnlo", "ns_coef", "hs_start", "hs_count", "hs_par", "hs_data", "hs_up", "hs_dn", "gam_start", "gam_count",
                             "gam_off", "gam_idx", "fn_start", "fn_count", "fn_slot")}  # fmt: skip
        t["tape_op"], t["tape_a"], t["tape_b"], t["tape_value"], slots = compile_bindings(trees, lambda name: set_by_name[name]["start"])
        self.blocks = []
        self.channel_blocks = [[] for _ in self.channels]
        cell = 0
        for ci, ch in enumerate(self.channels):
            for si in range(len(self.samples)):
                d = defined[ci][si]
                if not d:
                    continue
                n = ch.n_bins
                nominal = d[0]["data"]
                for kind in ("scal", "ns", "hs", "gam"):
                    t[f"{kind}_start"].append(len(t["gam_off" if kind == "gam" else f"{kind}_par"]))
                scal = []
                t["fn_start"].append(len(t["fn_slot"]))
                for name, type_ in modifiers:
                    mod = d[1].get(f"{type_}/{name}")
                    if not mod:
                        continue
                    if type_ == "purefunc":
                        t["fn_slot"].append(slots[name])
                        continue
                    start = set_by_name[name]["start"]
                    if type_ in ("normfactor", "lumi"):
                        scal.append(start)
                    elif type_ == "normsys":
                        hi, lo = mod["data"]["hi"], mod["data"]["lo"]
                        t["ns_par"].append(start)
                        t["ns_lnhi"].append(math.log(hi))
                        t["ns_lnlo"].append(math.log(lo))
                        t["ns_coef"].extend(code4_coefficients(hi, lo))
                    elif type_ == "histosys":
                        t["hs_par"].append(start)
                        t["hs_data"].append(len(t["hs_up"]))
                        t["hs_up"].extend(v - nom for v, nom in zip(mod["data"]["hi_data"], nominal))
                        t["hs_dn"].extend(nom - v for v, nom in zip(mod["data"]["lo_data"], nominal))
                    else:
                        t["gam_off"].append(len(t["gam_idx"]))
                        if type_ == "staterror":
                            t["gam_idx"].extend(stat_index[name][ch.offset : ch.offset + n].tolist())
                        else:
                            t["gam_idx"].extend(range(start, start + n))
                t["scal_par"].extend(scal)
                t["fn_count"].append(len(t["fn_slot"]) - t["fn_start"][-1])
                for kind in ("scal", "ns", "hs", "gam"):
                    t[f"{kind}_count"].append(len(t["gam_off" if kind == "gam" else f"{kind}_par"]) - t[f"{kind}_start"][-1])
                t["blk_offset"].append(ch.offset)
                t["blk_n"].append(n)
                t["blk_cell"].append(cell)
                t["nominal"].extend(nominal)
                block = SimpleNamespace(id=len(self.blocks), channel=ci, sample=si, offset=ch.offset, n=n, cell=cell, scal=scal)
                self.blocks.append(block)
                self.channel_blocks[ci].append(block)
                cell += n
        self.n_cells = cell

        # everything the kernel reads or writes; the arrays stay alive with the model
        t.update(
            gauss_idx=self.gauss_idx, gauss_aux=self.aux_index[self.gauss_idx], gauss_sigma=self.gauss_sigma,
            pois_idx=self.pois_idx, pois_aux=self.aux_index[self.pois_idx], pois_tau=self.pois_tau,
            nu=np.zeros(B), cell_nu=np.zeros(cell), cell_base=np.zeros(cell), cell_gamma=np.zeros(cell),
            block_factor=np.zeros(len(self.blocks)), cell_weight=np.zeros(cell),
            ns_val=np.zeros(len(t["ns_par"])), ns_der=np.zeros(len(t["ns_par"])),
            slot_val=np.zeros(len(t["tape_op"])), slot_adj=np.zeros(len(t["tape_op"])),
        )  # fmt: skip
        self._arrays = {name: np.ascontiguousarray(t[name], dtype=dtype) for name, dtype in _kernel.ARRAYS}
        self._struct = _kernel.HFModel(
            n_bins=B, n_pars=n_pars, n_blocks=len(self.blocks), n_cells=cell, n_gauss=len(gauss_idx), n_pois=len(pois_idx),
            n_slots=len(t["tape_op"]),
            **{name: a.ctypes.data for name, a in self._arrays.items()},
        )  # fmt: skip
        self._ref = _kernel.ctypes.addressof(self._struct)
        # scratch space filled by evaluate()
        self.nu = self._arrays["nu"]  # expected events per bin
        self.cell_nu = self._arrays["cell_nu"]  # per sample and bin
        self.cell_weight = self._arrays["cell_weight"]

    def evaluate(self, pars):
        """expected events for the given parameters; results are left in self.nu and self.cell_nu"""
        _lib.hf_evaluate(self._ref, _f64(pars).ctypes.data)
        return self.nu

    def accumulate(self, pars, grad, block_ids=None):
        """adds to grad the derivative of sum_cells cell_weight * cell_nu; evaluate(pars) must have been called before"""
        if block_ids is None:
            _lib.hf_accumulate(self._ref, pars.ctypes.data, grad.ctypes.data, None, 0)
        else:
            _lib.hf_accumulate(self._ref, pars.ctypes.data, grad.ctypes.data, block_ids.ctypes.data, len(block_ids))

    def expected_data(self, pars):
        """expected main and auxiliary data, as pyhf's Model.expected_data"""
        pars = _f64(pars)
        out = np.zeros(self.n_bins + self.n_aux)
        out[: self.n_bins] = self.evaluate(pars)
        out[self.n_bins + self.aux_index[self.gauss_idx]] = pars[self.gauss_idx]
        out[self.n_bins + self.aux_index[self.pois_idx]] = pars[self.pois_idx] * self.pois_tau
        return out

    def likelihood(self, data):
        return Likelihood(self, data)
