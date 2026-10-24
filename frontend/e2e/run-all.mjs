// Runs every e2e scenario (run.mjs) and the checks that compare scenarios: the head-roll sign (the portrait rotated
// 15 degrees clockwise must read ~30 degrees more roll than rotated counterclockwise) and two tracked people with a
// face each in the two-person scene. Exit 1 if anything fails. Writes .out/summary.json.
//
//   node e2e/run-all.mjs [--seconds 25] [--csp task|deploy]

import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const E2E = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const pass = [];
for (const k of ["--seconds", "--csp", "--delegate"]) {
  const i = args.indexOf(k);
  if (i >= 0) pass.push(k, args[i + 1]);
}
const scenarios = ["one", "two", "roll_cw", "roll_ccw"];
const reports = {};
let first = true;
for (const sc of scenarios) {
  const r = spawnSync(process.execPath, [path.join(E2E, "run.mjs"), "--scenario", sc, ...pass, ...(first ? [] : ["--no-build"])], {
    stdio: ["ignore", "ignore", "inherit"],
  });
  first = false;
  reports[sc] = JSON.parse(readFileSync(path.join(E2E, ".out", `report-${sc}.json`), "utf8"));
  reports[sc].exit = r.status;
}
const roll = (sc) => reports[sc].medians.head_roll_abs;
const lastPersons = reports.two.sample_video.at(-1)?.persons ?? [];
const cross = {
  roll_sign: roll("roll_cw") - roll("roll_ccw") > 20,
  two_people_two_faces: lastPersons.length === 2 && lastPersons.every((p) => p.face),
};
const summary = {
  ok: scenarios.every((sc) => reports[sc].ok) && Object.values(cross).every(Boolean),
  cross,
  scenarios: Object.fromEntries(scenarios.map((sc) => [sc, {
    ok: reports[sc].ok,
    failed: Object.entries(reports[sc].checks).filter(([, v]) => !v).map(([k]) => k),
    load_ms: reports[sc].metrics.load_ms,
    fps: reports[sc].metrics.fps && Math.round(reports[sc].metrics.fps * 10) / 10,
    video_ms: reports[sc].metrics.video_ms && Math.round(reports[sc].metrics.video_ms),
    backend: reports[sc].status.video_backend,
    max_people: reports[sc].metrics.max_people,
    roll: roll(sc),
  }])),
};
writeFileSync(path.join(E2E, ".out", "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
console.log(summary.ok ? "e2e all: OK" : "e2e all: FAILED");
process.exit(summary.ok ? 0 : 1);
