// Soak: run the real perception (run.mjs) for a long time and sample the memory of every Chromium process (page,
// GPU, and the renderer that hosts the vision worker) every 30 s. A leak (an unclosed ImageBitmap, a growing map)
// shows up as steady RSS growth. Writes .out/soak.json.
//
//   node e2e/soak.mjs [--minutes 15] [--scenario one]

import { spawn, execSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const E2E = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const MINUTES = Number(opt("--minutes", "15"));
const SCENARIO = opt("--scenario", "one");

/** Total RSS (MiB) of the Chromium processes started by this run. */
function chromeRssMiB() {
  try {
    const out = execSync("ps -eo rss=,args=", { encoding: "utf8" });
    return out.split("\n").filter((l) => /chrom(e|ium)/i.test(l) && /--use-fake-device-for-media-stream|--type=/.test(l))
      .reduce((n, l) => n + Number(l.trim().split(/\s+/)[0] || 0), 0) / 1024;
  } catch {
    return NaN;
  }
}

const child = spawn(process.execPath, [path.join(E2E, "run.mjs"), "--scenario", SCENARIO, "--seconds", String(MINUTES * 60)], {
  stdio: ["ignore", "ignore", "inherit"],
});
const samples = [];
const t0 = Date.now();
const timer = setInterval(() => samples.push({ s: Math.round((Date.now() - t0) / 1000), rss_mib: Math.round(chromeRssMiB()) }), 30000);
child.on("exit", (code) => {
  clearInterval(timer);
  const report = JSON.parse(readFileSync(path.join(E2E, ".out", `report-${SCENARIO}.json`), "utf8"));
  const settled = samples.filter((x) => x.s >= 120);
  const first = settled[0]?.rss_mib ?? NaN;
  const last = settled.at(-1)?.rss_mib ?? NaN;
  const summary = {
    minutes: MINUTES, scenario: SCENARIO, run_ok: report.ok, exit: code,
    fps: report.metrics.fps, video_observations: report.metrics.video_observations,
    audio_observations: report.metrics.audio_observations, counters: report.counters,
    main_thread_long_tasks: report.metrics.main_thread_long_tasks, page_errors: report.page_errors.length,
    rss_mib_after_2_min: first, rss_mib_end: last,
    rss_growth_mib_per_10_min: settled.length > 1 ? Math.round(((last - first) / ((settled.at(-1).s - settled[0].s) / 600)) * 10) / 10 : null,
    samples,
  };
  writeFileSync(path.join(E2E, ".out", "soak.json"), `${JSON.stringify(summary, null, 2)}\n`);
  console.log(JSON.stringify({ ...summary, samples: undefined }, null, 2));
});
