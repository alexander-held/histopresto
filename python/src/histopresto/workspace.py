"""Reading and checking a pyhf workspace (JSON)."""

import json
import math

MODIFIER_TYPES = ["histosys", "lumi", "normfactor", "normsys", "shapefactor", "shapesys", "staterror", "purefunc"]


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_number_array(v, n=None):
    return isinstance(v, list) and all(_is_number(x) for x in v) and (n is None or len(v) == n)


def parse_workspace(source):
    """Returns {channels, observations: {name: data}, measurements, bindings}; raises ValueError listing what is wrong."""
    ws = source
    if isinstance(source, (str, bytes)):
        try:
            ws = json.loads(source)
        except json.JSONDecodeError as e:
            raise ValueError(f"This is not valid JSON: {e}") from None
    problems = []
    bad = problems.append
    if not isinstance(ws, dict) or not isinstance(ws.get("channels"), list) or not ws["channels"]:
        raise ValueError('This does not look like a pyhf workspace: there is no "channels" list.')
    if not isinstance(ws.get("observations"), list):
        raise ValueError(
            'This does not look like a pyhf workspace: there is no "observations" list. Model-only specifications without data are not supported.'
        )

    def first_data(channel):
        samples = channel.get("samples")
        first = samples[0] if isinstance(samples, list) and samples else None
        data = first.get("data") if isinstance(first, dict) else None
        return data if isinstance(data, list) else []

    seen_channels = set()
    for channel in ws["channels"]:
        cn = channel.get("name") if isinstance(channel, dict) else None
        if not isinstance(cn, str):
            bad("a channel has no name")
            continue
        if cn in seen_channels:
            bad(f'channel "{cn}" appears more than once')
        seen_channels.add(cn)
        if not isinstance(channel.get("samples"), list) or not channel["samples"]:
            bad(f'channel "{cn}" has no samples')
            continue
        n_bins = len(first_data(channel))
        seen_samples = set()
        for sample in channel["samples"]:
            sn = sample.get("name") if isinstance(sample, dict) else None
            where = f'sample "{sn}" in channel "{cn}"'
            if not isinstance(sn, str):
                bad(f'a sample in channel "{cn}" has no name')
                continue
            if sn in seen_samples:
                bad(f"{where} appears more than once")
            seen_samples.add(sn)
            data = sample.get("data")
            if not _is_number_array(data) or not data:
                bad(f'{where} has no numeric "data"')
                continue
            if len(data) != n_bins:
                bad(f"{where} has {len(data)} bins, other samples in this channel have {n_bins}")
            if not isinstance(sample.get("modifiers"), list):
                bad(f'{where} has no "modifiers" list')
                continue
            n = len(data)
            for mod in sample["modifiers"]:
                ok = isinstance(mod, dict)
                label = f'modifier "{mod.get("name") if ok else None}" ({mod.get("type") if ok else None}) of {where}'
                if not ok or not isinstance(mod.get("name"), str) or mod.get("type") not in MODIFIER_TYPES:
                    bad(f"{label} has an unknown type; supported are {', '.join(MODIFIER_TYPES)}")
                    continue
                d = mod.get("data")
                if mod["type"] == "normsys":
                    if not isinstance(d, dict) or not _is_number(d.get("hi")) or not _is_number(d.get("lo")):
                        bad(f"{label} needs data {{hi, lo}}")
                elif mod["type"] == "histosys":
                    if not isinstance(d, dict) or not _is_number_array(d.get("hi_data"), n) or not _is_number_array(d.get("lo_data"), n):
                        bad(f"{label} needs data {{hi_data, lo_data}} with {n} bins each")
                elif mod["type"] in ("staterror", "shapesys"):
                    if not _is_number_array(d, n):
                        bad(f"{label} needs a list of {n} uncertainties as data")

    observations = {}
    for obs in ws["observations"]:
        if not isinstance(obs, dict) or not isinstance(obs.get("name"), str) or not _is_number_array(obs.get("data")):
            bad("an observation has no name or no numeric data")
        else:
            observations[obs["name"]] = obs["data"]
    for channel in ws["channels"]:
        if not isinstance(channel, dict) or not isinstance(channel.get("name"), str):
            continue
        if not isinstance(channel.get("samples"), list) or not channel["samples"]:
            continue
        obs = observations.get(channel["name"])
        n_bins = len(first_data(channel))
        if obs is None:
            bad(f'there is no observation for channel "{channel["name"]}"')
        elif len(obs) != n_bins:
            bad(f'the observation for channel "{channel["name"]}" has {len(obs)} bins, the samples have {n_bins}')

    raw = ws.get("measurements")
    if not isinstance(raw, list) or not raw:
        raw = [{"name": "default", "config": {"poi": "", "parameters": []}}]
    measurements = []
    for i, m in enumerate(raw):
        m = m if isinstance(m, dict) else {}
        config = m.get("config") if isinstance(m.get("config"), dict) else {}
        parameters = config.get("parameters") if isinstance(config.get("parameters"), list) else []
        for p in parameters:
            if not isinstance(p, dict) or not isinstance(p.get("name"), str):
                bad(f"a parameter configuration in measurement {i} has no name")
        measurements.append(
            {
                "name": m["name"] if isinstance(m.get("name"), str) else f"measurement {i}",
                "config": {"poi": config["poi"] if isinstance(config.get("poi"), str) else "", "parameters": parameters},
            }
        )

    # expressions for purefunc modifiers
    bindings = ws.get("bindings", [])
    if not isinstance(bindings, list):
        bad('"bindings" is not a list')
    else:
        for b in bindings:
            ok = isinstance(b, dict) and isinstance(b.get("names"), list) and b["names"] and all(isinstance(n, str) for n in b["names"])
            if not ok or not isinstance(b.get("expression"), str) or not isinstance(b.get("language"), str):
                bad('a binding needs a list of "names", an "expression" and a "language"')

    if problems:
        shown = [f"• {p}" for p in problems[:8]]
        if len(problems) > 8:
            shown.append(f"• … and {len(problems) - 8} more")
        raise ValueError("The workspace has problems:\n" + "\n".join(shown))
    return {"channels": ws["channels"], "observations": observations, "measurements": measurements, "bindings": bindings}
