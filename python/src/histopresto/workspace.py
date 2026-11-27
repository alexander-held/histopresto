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


def combine_workspaces(parts):
    """Combines parsed workspaces into one, like pyhf's Workspace.combine: every channel may be in one workspace only.

    Measurements with the same name are merged, as are identical bindings; where workspaces configure a parameter
    or the POI differently, the first one counts and a note says so.
    parts: [{name, workspace}], name is only for messages; returns a workspace like parse_workspace, with a list of notes.
    """
    if len(parts) == 1:
        return {**parts[0]["workspace"], "notes": []}
    channels = []
    observations = {}
    owner = {}
    for part in parts:
        name, workspace = part["name"], part["workspace"]
        for channel in workspace["channels"]:
            cn = channel["name"]
            if cn in owner:
                raise ValueError(f'Channel "{cn}" is in both {owner[cn]} and {name}; workspaces can only be combined if their channels differ.')
            owner[cn] = name
            channels.append(channel)
            observations[cn] = workspace["observations"][cn]

    def canonical(v):
        return json.dumps(v, sort_keys=True)

    merged = {}  # measurement name -> {poi, parameters, differing, pois}
    for part in parts:
        for m in part["workspace"]["measurements"]:
            into = merged.setdefault(m["name"], {"poi": "", "parameters": {}, "differing": [], "pois": []})
            poi = m["config"]["poi"]
            if poi:
                if poi not in into["pois"]:
                    into["pois"].append(poi)
                if not into["poi"]:
                    into["poi"] = poi
            for p in m["config"]["parameters"]:
                known = into["parameters"].setdefault(p["name"], p)
                if canonical(known) != canonical(p) and p["name"] not in into["differing"]:
                    into["differing"].append(p["name"])
    notes = []
    measurements = []
    for name, m in merged.items():
        measurements.append({"name": name, "config": {"poi": m["poi"], "parameters": list(m["parameters"].values())}})
        if len(m["pois"]) > 1:
            notes.append(f'The workspaces name different parameters of interest for measurement "{name}" ({", ".join(m["pois"])}); {m["poi"]} is used.')
        n = len(m["differing"])
        if n:
            shown = ", ".join(m["differing"][:10]) + (f" and {n - 10} more" if n > 10 else "")
            notes.append(
                f'The workspaces configure {"a parameter" if n == 1 else f"{n} parameters"} of measurement "{name}" differently ({shown}); '
                "the settings of the workspace loaded first are used."
            )
    if len(measurements) > 1:
        notes.append("The workspaces have measurements with different names, which are not merged: each one only has the settings of the workspaces that define it.")

    bindings = []
    bound = {}
    for part in parts:
        for b in part["workspace"]["bindings"]:
            key = canonical(b)
            if all(n in bound and bound[n][0] == key for n in b["names"]):
                continue
            clash = next((n for n in b["names"] if n in bound), None)
            if clash is not None:
                raise ValueError(f'The binding "{clash}" is defined differently in {bound[clash][1]} and {part["name"]}.')
            for n in b["names"]:
                bound[n] = (key, part["name"])
            bindings.append(b)
    return {"channels": channels, "observations": observations, "measurements": measurements, "bindings": bindings, "notes": notes}
