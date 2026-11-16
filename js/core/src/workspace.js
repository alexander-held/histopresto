// Reading and checking a pyhf workspace (JSON).

const MODIFIER_TYPES = ["histosys", "lumi", "normfactor", "normsys", "shapefactor", "shapesys", "staterror"];
const isNumber = (v) => typeof v === "number" && Number.isFinite(v);
const isNumberArray = (v, n) => Array.isArray(v) && v.every(isNumber) && (n === undefined || v.length === n);

// returns {channels, observations: {name: data}, measurements}; throws an Error listing what is wrong
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

  if (problems.length) {
    const shown = problems.slice(0, 8).map((p) => `• ${p}`);
    if (problems.length > 8) shown.push(`• … and ${problems.length - 8} more`);
    throw new Error(`The workspace has problems:\n${shown.join("\n")}`);
  }
  return { channels: ws.channels, observations, measurements };
}

export { MODIFIER_TYPES, parseWorkspace };
