// Contours of a function sampled on a grid: z[i][j] = z(x[i], y[j]).

const BIG = 1e30;

// finer grid by cubic (Catmull-Rom) interpolation, exact for quadratic surfaces; x and y evenly spaced
function refine(x, y, z, factor = 4) {
  if (x.length < 3 || y.length < 3 || z.some((row) => row.some((v) => !Number.isFinite(v)))) return { x, y, z };
  const up = (v) => {
    const n = v.length;
    const at = (i) => (i < 0 ? 3 * v[0] - 3 * v[1] + v[2] : i >= n ? 3 * v[n - 1] - 3 * v[n - 2] + v[n - 3] : v[i]);
    const out = [];
    for (let i = 0; i < n - 1; i++) {
      const [a, b, c, d] = [at(i - 1), v[i], v[i + 1], at(i + 2)];
      for (let s = 0; s < factor; s++) {
        const t = s / factor;
        out.push(b + 0.5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a))));
      }
    }
    out.push(v[n - 1]);
    return out;
  };
  const rows = z.map(up);
  const fine = rows[0].map(() => []);
  rows[0].forEach((_, j) => up(rows.map((row) => row[j])).forEach((v, i) => (fine[j][i] = v)));
  // fine is indexed [j][i]: transpose back
  const X = up(x);
  return { x: X, y: up(y), z: X.map((_, i) => fine.map((col) => col[i])) };
}

// Lines along which z equals the level, as polylines [[x, y], ...] (marching squares). With closed,
// the region below the level is closed along the border of the grid, so that it can be filled.
function contour(x, y, z, level, closed = false) {
  if (closed) {
    x = [x[0], ...x, x[x.length - 1]];
    y = [y[0], ...y, y[y.length - 1]];
    const edge = y.map(() => BIG);
    z = [edge, ...z.map((row) => [BIG, ...row, BIG]), edge];
  }
  const nx = x.length;
  const ny = y.length;
  const val = (i, j) => (Number.isFinite(z[i][j]) ? z[i][j] : BIG);
  const points = new Map(); // crossing of a grid edge, by the two nodes of the edge
  const links = new Map();
  const cross = (i0, j0, i1, j1) => {
    const key = `${i0 * ny + j0}:${i1 * ny + j1}`;
    if (!points.has(key)) {
      const [a, b] = [val(i0, j0), val(i1, j1)];
      const t = (level - a) / (b - a);
      points.set(key, [x[i0] + t * (x[i1] - x[i0]), y[j0] + t * (y[j1] - y[j0])]);
      links.set(key, []);
    }
    return key;
  };
  const join = (p, q) => (links.get(p).push(q), links.get(q).push(p));
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      const inside = [val(i, j), val(i + 1, j), val(i + 1, j + 1), val(i, j + 1)].map((v) => v < level);
      // crossings on the bottom, right, top and left edge of the cell
      const e = [
        inside[0] !== inside[1] && cross(i, j, i + 1, j),
        inside[1] !== inside[2] && cross(i + 1, j, i + 1, j + 1),
        inside[3] !== inside[2] && cross(i, j + 1, i + 1, j + 1),
        inside[0] !== inside[3] && cross(i, j, i, j + 1),
      ];
      const hit = e.filter(Boolean);
      if (hit.length === 2) join(hit[0], hit[1]);
      else if (hit.length === 4) {
        // saddle: the value in the middle of the cell decides which corners are connected
        const middle = (val(i, j) + val(i + 1, j) + val(i + 1, j + 1) + val(i, j + 1)) / 4 < level;
        if (middle === inside[0]) (join(e[0], e[1]), join(e[2], e[3]));
        else (join(e[3], e[0]), join(e[1], e[2]));
      }
    }
  }
  // chain the pieces: open lines from their ends first, the rest are loops
  const lines = [];
  const seen = new Set();
  const walk = (start) => {
    const line = [points.get(start)];
    seen.add(start);
    let at = start;
    for (;;) {
      const next = links.get(at).find((q) => !seen.has(q));
      if (next === undefined) break;
      seen.add(next);
      line.push(points.get(next));
      at = next;
    }
    if (links.get(at).includes(start) && line.length > 2) line.push(line[0]);
    lines.push(line);
  };
  for (const [key, l] of links) if (l.length === 1 && !seen.has(key)) walk(key);
  for (const key of links.keys()) if (!seen.has(key)) walk(key);
  return lines;
}

export { contour, refine };
