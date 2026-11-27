// Website only: normsys factors of zero are replaced, with a warning.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { Model, parseWorkspace } from "@histopresto/core";
import { repairNormsys } from "../../js/app/src/repair.js";

test("normsys with a factor of zero", () => {
  const raw = JSON.parse(fs.readFileSync(new URL("../reference/workspaces/medium.json", import.meta.url), "utf8"));
  assert.deepEqual(repairNormsys(parseWorkspace(raw)), []);
  const mod = raw.channels[0].samples.flatMap((s) => s.modifiers).find((m) => m.type === "normsys");
  mod.data = { hi: mod.data.hi, lo: 0 };
  const ws = parseWorkspace(raw);
  const broken = new Model(ws, 0);
  assert.ok(!Number.isFinite(broken.likelihood(broken.observed).value(broken.init)));
  const notes = repairNormsys(ws);
  assert.equal(notes.length, 1);
  assert.match(notes[0], new RegExp(`ill-defined.*${mod.name} \\(down 0\\)`));
  assert.equal(mod.data.lo, 0); // the source is left as it was
  const model = new Model(ws, 0);
  assert.ok(Number.isFinite(model.likelihood(model.observed).value(model.init)));
});
