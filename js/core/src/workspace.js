// Reading and checking a pyhf workspace (JSON).

const MODIFIER_TYPES = ["histosys", "lumi", "normfactor", "normsys", "shapefactor", "shapesys", "staterror", "purefunc"];
const isNumber = (v) => typeof v === "number" && Number.isFinite(v);
const isNumberArray = (v, n) => Array.isArray(v) && v.every(isNumber) && (n === undefined || v.length === n);

// returns {channels, observations: {name: data}, measurements, bindings}; throws an Error listing what is wrong
function parseWorkspace(input) {
  let ws = input;
  if (typeof input === "string") {
    try {
      ws = JSON.parse(input);
    } catch (e) {
      throw new Error(`This is not valid JSON: ${e.message}`);
    }
  }
  const problems = [];
  const bad = (msg) => problems.push(msg);
  if (!ws || typeof ws !== "object" || !Array.isArray(ws.channels) || ws.channels.length === 0) {
    throw new Error('This does not look like a pyhf workspace: there is no "channels" list.');
  }
  if (!Array.isArray(ws.observations)) {
    throw new Error('This does not look like a pyhf workspace: there is no "observations" list. Model-only specifications without data are not supported.');
  }

  const seenChannels = new Set();
  for (const channel of ws.channels) {
    const cn = channel && channel.name;
    if (typeof cn !== "string") {
      bad("a channel has no name");
      continue;
    }
    if (seenChannels.has(cn)) bad(`channel "${cn}" appears more than once`);
    seenChannels.add(cn);
    if (!Array.isArray(channel.samples) || channel.samples.length === 0) {
      bad(`channel "${cn}" has no samples`);
      continue;
    }
    const nBins = Array.isArray(channel.samples[0].data) ? channel.samples[0].data.length : 0;
    const seenSamples = new Set();
    for (const sample of channel.samples) {
      const sn = sample && sample.name;
      const where = `sample "${sn}" in channel "${cn}"`;
      if (typeof sn !== "string") {
        bad(`a sample in channel "${cn}" has no name`);
        continue;
      }
      if (seenSamples.has(sn)) bad(`${where} appears more than once`);
      seenSamples.add(sn);
      if (!isNumberArray(sample.data) || sample.data.length === 0) {
        bad(`${where} has no numeric "data"`);
        continue;
      }
      if (sample.data.length !== nBins) bad(`${where} has ${sample.data.length} bins, other samples in this channel have ${nBins}`);
      if (!Array.isArray(sample.modifiers)) {
        bad(`${where} has no "modifiers" list`);
        continue;
      }
      const n = sample.data.length;
      for (const mod of sample.modifiers) {
        const label = `modifier "${mod && mod.name}" (${mod && mod.type}) of ${where}`;
        if (!mod || typeof mod.name !== "string" || !MODIFIER_TYPES.includes(mod.type)) {
          bad(`${label} has an unknown type; supported are ${MODIFIER_TYPES.join(", ")}`);
        } else if (mod.type === "normsys") {
          if (!mod.data || !isNumber(mod.data.hi) || !isNumber(mod.data.lo)) bad(`${label} needs data {hi, lo}`);
        } else if (mod.type === "histosys") {
          if (!mod.data || !isNumberArray(mod.data.hi_data, n) || !isNumberArray(mod.data.lo_data, n)) {
            bad(`${label} needs data {hi_data, lo_data} with ${n} bins each`);
          }
        } else if (mod.type === "staterror" || mod.type === "shapesys") {
          if (!isNumberArray(mod.data, n)) bad(`${label} needs a list of ${n} uncertainties as data`);
        }
      }
    }
  }

  const observations = {};
  for (const obs of ws.observations) {
    if (!obs || typeof obs.name !== "string" || !isNumberArray(obs.data)) bad("an observation has no name or no numeric data");
    else observations[obs.name] = obs.data;
  }
  for (const channel of ws.channels) {
    if (typeof channel.name !== "string" || !Array.isArray(channel.samples) || !channel.samples[0]) continue;
    const obs = observations[channel.name];
    const nBins = (channel.samples[0].data || []).length;
    if (!obs) bad(`there is no observation for channel "${channel.name}"`);
    else if (obs.length !== nBins) bad(`the observation for channel "${channel.name}" has ${obs.length} bins, the samples have ${nBins}`);
  }

  let measurements = ws.measurements;
  if (!Array.isArray(measurements) || measurements.length === 0) {
    measurements = [{ name: "default", config: { poi: "", parameters: [] } }];
  }
  measurements = measurements.map((m, i) => {
    const config = (m && m.config) || {};
    const parameters = Array.isArray(config.parameters) ? config.parameters : [];
    for (const p of parameters) {
      if (!p || typeof p.name !== "string") bad(`a parameter configuration in measurement ${i} has no name`);
    }
    return {
      name: m && typeof m.name === "string" ? m.name : `measurement ${i}`,
      config: { poi: typeof config.poi === "string" ? config.poi : "", parameters },
    };
  });

  // expressions for purefunc modifiers
  const bindings = ws.bindings === undefined ? [] : ws.bindings;
  if (!Array.isArray(bindings)) bad('"bindings" is not a list');
  else {
    for (const b of bindings) {
      const ok = b && Array.isArray(b.names) && b.names.length > 0 && b.names.every((n) => typeof n === "string");
      if (!ok || typeof b.expression !== "string" || typeof b.language !== "string") bad('a binding needs a list of "names", an "expression" and a "language"');
    }
  }

  if (problems.length) {
    const shown = problems.slice(0, 8).map((p) => `• ${p}`);
    if (problems.length > 8) shown.push(`• … and ${problems.length - 8} more`);
    throw new Error(`The workspace has problems:\n${shown.join("\n")}`);
  }
  return { channels: ws.channels, observations, measurements, bindings };
}

// Combines parsed workspaces into one, like pyhf's Workspace.combine: every channel may be in one workspace only.
// Measurements with the same name are merged, as are identical bindings; where workspaces configure a parameter
// or the POI differently, the first one counts and a note says so.
// parts: [{name, workspace}], name is only for messages; returns a workspace like parseWorkspace, with a list of notes
function combineWorkspaces(parts) {
  if (parts.length === 1) return { ...parts[0].workspace, notes: [] };
  const channels = [];
  const observations = {};
  const owner = {};
  for (const { name, workspace } of parts) {
    for (const channel of workspace.channels) {
      if (channel.name in owner) throw new Error(`Channel "${channel.name}" is in both ${owner[channel.name]} and ${name}; workspaces can only be combined if their channels differ.`);
      owner[channel.name] = name;
      channels.push(channel);
      observations[channel.name] = workspace.observations[channel.name];
    }
  }

  const canonical = (v) => JSON.stringify(v, (k, x) => (x && typeof x === "object" && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : 1))) : x));
  const merged = new Map(); // measurement name -> {poi, parameters: Map, differing: Set, pois: Set}
  for (const { workspace } of parts) {
    for (const m of workspace.measurements) {
      if (!merged.has(m.name)) merged.set(m.name, { poi: "", parameters: new Map(), differing: new Set(), pois: new Set() });
      const into = merged.get(m.name);
      if (m.config.poi) {
        into.pois.add(m.config.poi);
        if (!into.poi) into.poi = m.config.poi;
      }
      for (const p of m.config.parameters) {
        const known = into.parameters.get(p.name);
        if (!known) into.parameters.set(p.name, p);
        else if (canonical(known) !== canonical(p)) into.differing.add(p.name);
      }
    }
  }
  const notes = [];
  const measurements = [];
  for (const [name, m] of merged) {
    measurements.push({ name, config: { poi: m.poi, parameters: [...m.parameters.values()] } });
    if (m.pois.size > 1) notes.push(`The workspaces name different parameters of interest for measurement "${name}" (${[...m.pois].join(", ")}); ${m.poi} is used.`);
    if (m.differing.size) {
      const shown = [...m.differing].slice(0, 10).join(", ") + (m.differing.size > 10 ? ` and ${m.differing.size - 10} more` : "");
      notes.push(`The workspaces configure ${m.differing.size === 1 ? "a parameter" : `${m.differing.size} parameters`} of measurement "${name}" differently (${shown}); the settings of the workspace loaded first are used.`);
    }
  }
  if (measurements.length > 1) notes.push(`The workspaces have measurements with different names, which are not merged: each one only has the settings of the workspaces that define it.`);

  const bindings = [];
  const bound = {};
  for (const { name, workspace } of parts) {
    for (const b of workspace.bindings) {
      const key = canonical(b);
      const fresh = b.names.filter((n) => bound[n] === undefined);
      if (fresh.length === 0 && b.names.every((n) => bound[n].key === key)) continue;
      const clash = b.names.find((n) => bound[n] !== undefined);
      if (clash !== undefined) throw new Error(`The binding "${clash}" is defined differently in ${bound[clash].name} and ${name}.`);
      for (const n of b.names) bound[n] = { key, name };
      bindings.push(b);
    }
  }
  return { channels, observations, measurements, bindings, notes };
}

export { MODIFIER_TYPES, parseWorkspace, combineWorkspaces };
