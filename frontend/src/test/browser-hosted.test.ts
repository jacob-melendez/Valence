// @vitest-environment-options {"url": "https://valence-web.example/?utm_source=devpost"}
import { act, cleanup, render } from "@testing-library/react";
import { createElement as h } from "react";
import { afterEach, describe, expect, it } from "vitest";

import { boot, getBrowserMode, getState, isHostedSite, setSource } from "@/lib/valence";
import { Route } from "@/routes/index";

// The hosted site (any non-local hostname): a shared link with tracking parameters still lands on the tour, and only
// the console's own parameters opt out of it.
describe("hosted Valence Web", () => {
  afterEach(() => { cleanup(); delete (navigator as { mediaDevices?: unknown }).mediaDevices; });

  it("only source, demo, api and perception opt a hosted visit out of the tour", () => {
    expect(window.location.hostname).toBe("valence-web.example");
    const at = (q: string) => { window.history.replaceState(null, "", `/${q}`); return isHostedSite(); };
    for (const q of ["", "?utm_source=devpost", "?fbclid=IwAR123", "?ref=x", "?delegate=GPU"]) expect([q, at(q)]).toEqual([q, true]);
    for (const q of ["?source=browser", "?demo=tour", "?api=http://127.0.0.1:8000", "?perception=mock", "?demo=x&utm_source=y"])
      expect([q, at(q)]).toEqual([q, false]);
    window.history.replaceState(null, "", "/?utm_source=devpost");
  });

  it("a ?utm_source visit boots the tour without asking for the camera", async () => {
    const calls: unknown[] = [];
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: {
      getUserMedia: (c: unknown) => { calls.push(c); return new Promise(() => {}); },
    } });
    const Index = (Route as unknown as { options: { component: () => React.ReactElement } }).options.component;
    await act(async () => { render(h(Index)); });
    await act(async () => { await new Promise((r) => setTimeout(r, 50)); });
    boot(); // once only: the page's boot already ran
    expect([getState().source, getBrowserMode()]).toEqual(["browser", "tour"]);
    expect(calls).toEqual([]);
    setSource("mock");
  });
});
