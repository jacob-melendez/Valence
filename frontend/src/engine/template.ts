/**
 * Instant template explanation, sent with every prediction before the LLM narration arrives
 * (valence/explain/template.py).
 *
 * It also serves as the narrator's last-resort fallback, so it obeys the same rules: only numbers that appear
 * in the fact sheet's display strings, masking language only when the override fired, room drivers phrased as
 * the room, people referred to as "Person #N". The opener is chosen by SHA-1 of the event id, so the browser
 * and the Python engine pick the same sentence for the same event.
 */
import { STATE_DISPLAY, type FactSheet } from "./contracts";
import { topDrivers } from "./explainer";
import { sha1Hex } from "./sha";

// ---------- Constants ----------

/** Matches the served model's confidence shrink rule (valence/explain/template.py:16). */
export const OOD_NOTE_AFTER = 3;

/** valence/explain/template.py:18, in Python str.format syntax. */
export const OPENERS = [
  "Model predicted {label} ({conf}) for {person}.",
  "{person} reads as {label} with {conf} confidence.",
  "{label} ({conf}) is the most likely state for {person}.",
] as const;

// ---------- Helpers ----------

/** True when `s` holds an unpaired UTF-16 surrogate, a code point `str.encode()` refuses. */
function hasLoneSurrogate(s: string): boolean {
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      const d = s.charCodeAt(i + 1); // NaN past the end
      if (d >= 0xdc00 && d <= 0xdfff) i++;
      else return true;
    } else if (c >= 0xdc00 && c <= 0xdfff) return true;
  }
  return false;
}

/**
 * `options[int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(options)]` (valence/explain/template.py:25).
 * sha1Hex hashes the UTF-8 bytes of the string, like `str.encode()`. A lone surrogate throws, as Python's
 * strict UTF-8 encoder does (sha1Hex itself would hash it as U+FFFD and pick an opener Python never would).
 */
export function pick<T>(options: readonly T[], key: string): T {
  if (hasLoneSurrogate(key))
    throw new RangeError("UnicodeEncodeError: 'utf-8' codec can't encode a surrogate");
  const i = BigInt(`0x${sha1Hex(key)}`) % BigInt(options.length);
  return options[Number(i)] as T;
}

/** Python `str.format` with named fields only (`{label}`); values are inserted verbatim, never re-parsed. */
function format(template: string, fields: Readonly<Record<string, string>>): string {
  return template.replace(/\{(\w+)\}/g, (_, name: string) => {
    const v = fields[name];
    if (v === undefined) throw new Error(`KeyError: ${JSON.stringify(name)}`);
    return v;
  });
}

/** "a", "a and b", "a, b and c" (valence/explain/template.py:29). */
function join(parts: readonly string[]): string {
  if (parts.length <= 1) return parts.join("");
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1] as string}`;
}

// ---------- Template ----------

/** The template explanation for a fact sheet (valence/explain/template.py:35 template_text). */
export function templateText(sheet: FactSheet): string {
  const opener = format(pick(OPENERS, sheet.event_id), {
    label: sheet.label_display,
    conf: sheet.confidence_display,
    person: sheet.person,
  });
  const contextLayers = sheet.layers.filter((layer) => layer.name !== "Facial Surface");
  const drivers = topDrivers(
    sheet.mask_override || !sheet.face_visible ? contextLayers : sheet.layers,
    3,
  );
  const evidence = join(drivers.map((d) => d.display));
  let body: string;
  if (sheet.mask_override && sheet.face_only.label === "Collaborative_Happy") {
    const faceName = STATE_DISPLAY[sheet.face_only.label] ?? sheet.face_only.label;
    body =
      `The face alone suggests ${faceName}, but ${evidence} point the other way. ` +
      "Context corrected the facial mask.";
  } else if (!sheet.face_visible) {
    body = `The face is not readable right now, so the call rests on context: ${evidence}.`;
  } else if (evidence) {
    body = `Main evidence: ${evidence}.`;
  } else {
    body = "The window is still filling, so evidence is limited.";
  }
  if (sheet.data_quality.ood_features.length > OOD_NOTE_AFTER) {
    body += " Some readings are outside the training range, so confidence is reduced.";
  }
  return `${opener} ${body}`;
}
