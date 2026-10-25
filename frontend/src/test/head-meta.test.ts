import { describe, expect, it } from "vitest";

import { Route as RootRoute } from "@/routes/__root";
import { Route as IndexRoute } from "@/routes/index";

// The page's title, description and social preview are prerendered into the static shell (link unfurls, search
// results): one source in __root.tsx, in the pitch's wording (signals, simulated data), with no template leftovers.
type Meta = { title?: string; name?: string; property?: string; content?: string };
type HeadFn = () => { meta?: Meta[] };
const headOf = (r: unknown) => (r as { options: { head?: HeadFn } }).options.head;

describe("page metadata", () => {
  it("the index route sets no head of its own (the root's is the only one)", () => {
    expect(headOf(IndexRoute)).toBeUndefined();
  });

  it("the root head names Valence Web, says simulated, and carries no template or stale wording", () => {
    const meta = headOf(RootRoute)?.().meta ?? [];
    const text = meta.flatMap((m) => [m.title, m.content]).filter((v): v is string => typeof v === "string").join("\n");
    expect(text).toContain("Valence Web");
    expect(text).toContain("simulated");
    expect(text).not.toMatch(/Lovable|Valence Console|not faces|offline/i);
    const description = meta.find((m) => m.name === "description")?.content ?? "";
    expect(description.slice(0, 150)).toMatch(/signals/);
    expect(description.slice(0, 150)).toMatch(/simulated/);
    expect(meta.find((m) => m.title)?.title).toMatch(/^Valence Web/);
  });
});
