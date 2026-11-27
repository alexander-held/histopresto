// Repairs of workspaces that the website applies before building the model, each with a warning.

const NORMSYS_FLOOR = 1e-3;

// normsys factors of zero or below: the interpolation uses their logarithm, so the model is not defined.
// They are replaced by a factor of 0.001. Changes the workspace (as parsed) in place and returns the warnings.
function repairNormsys(ws) {
  const cases = [];
  for (const channel of ws.channels) {
    for (const sample of channel.samples) {
      sample.modifiers = sample.modifiers.map((mod) => {
        if (mod.type !== "normsys" || (mod.data.hi > 0 && mod.data.lo > 0)) return mod;
        const { hi, lo } = mod.data;
        const which = [!(hi > 0) && `up ${hi}`, !(lo > 0) && `down ${lo}`].filter(Boolean).join(", ");
        cases.push(`${mod.name} (${which}) for ${sample.name} in ${channel.name}`);
        return { ...mod, data: { hi: hi > 0 ? hi : NORMSYS_FLOOR, lo: lo > 0 ? lo : NORMSYS_FLOOR } };
      });
    }
  }
  if (!cases.length) return [];
  const shown = cases.slice(0, 5).join("; ") + (cases.length > 5 ? `; and ${cases.length - 5} more` : "");
  return [
    `This workspace is ill-defined: ${cases.length === 1 ? "a normsys modifier has a factor" : `${cases.length} normsys modifiers have factors`} of zero or below (${shown}). ` +
      `The interpolation between the variations takes the logarithm of these factors, so the model cannot be evaluated as written. ` +
      `As a fix, each such factor is replaced by ${NORMSYS_FLOOR}, which scales the sample down by a factor of 1000; results depend on this choice.`,
  ];
}

export { repairNormsys };
