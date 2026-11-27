// Combining workspaces: the combination of the two halves of a workspace is the workspace itself.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { Model, parseWorkspace, combineWorkspaces } from "@histopresto/core";

const read = (name) => JSON.parse(fs.readFileSync(new URL(`../reference/workspaces/${name}.json`, import.meta.url), "utf8"));
const part = (ws, keep, name) => ({
  name,
  workspace: parseWorkspace({ ...ws, channels: ws.channels.filter((c, i) => keep(i)), observations: ws.observations.filter((o) => keep(ws.channels.findIndex((c) => c.name === o.name))) }),
});

for (const name of ["medium", "purefunc_mixed"]) {
  test(`split and combined: ${name}`, () => {
    const ws = read(name);
    assert.ok(ws.channels.length > 1);
    const whole = new Model(parseWorkspace(ws), 0);
    const combined = combineWorkspaces([part(ws, (i) => i === 0, "a"), part(ws, (i) => i > 0, "b")]);
    assert.deepEqual(combined.notes, []);
    const model = new Model(combined, 0);
    assert.deepEqual(model.names, whole.names);
    assert.deepEqual(model.init, whole.init);
    assert.equal(model.likelihood(model.observed).value(model.init), whole.likelihood(whole.observed).value(whole.init));
  });
}

test("conflicts", () => {
  const ws = read("medium");
  const a = part(ws, (i) => i === 0, "a");
  assert.throws(() => combineWorkspaces([a, a]), /is in both a and a/);
  const b = part(ws, (i) => i > 0, "b");
  const poi = a.workspace.measurements[0].config.poi;
  a.workspace.measurements[0].config.parameters = [{ name: poi, inits: [1], bounds: [[0, 5]] }];
  b.workspace.measurements[0].config.parameters = [{ bounds: [[0, 5]], name: poi, inits: [1] }, { name: "other", fixed: true }];
  assert.deepEqual(combineWorkspaces([a, b]).notes, []);
  b.workspace.measurements[0].config.parameters[0].inits = [2];
  const combined = combineWorkspaces([a, b]);
  assert.equal(combined.notes.length, 1);
  // a workspace without measurements adds none; measurements with different names are kept, with a note
  const bare = { name: "c", workspace: { ...b.workspace, measurements: parseWorkspace({ ...ws, measurements: undefined }).measurements } };
  assert.deepEqual(combineWorkspaces([bare, a]).measurements, a.workspace.measurements);
  assert.deepEqual(combineWorkspaces([bare, a]).notes, []);
  const other = { name: "d", workspace: { ...b.workspace, measurements: [{ name: "other", config: { poi: poi, parameters: [] } }] } };
  const two = combineWorkspaces([a, other]);
  assert.deepEqual(two.measurements.map((m) => m.name), [a.workspace.measurements[0].name, "other"]);
  assert.equal(two.notes.length, 1);
  assert.deepEqual(combined.measurements[0].config.parameters, [{ name: poi, inits: [1], bounds: [[0, 5]] }, { name: "other", fixed: true }]);
});
