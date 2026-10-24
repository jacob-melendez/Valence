import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { boot, useValence } from "@/lib/valence";
import {
  AffectMap, CameraView, Welcome, Controls, Evaluation, Governance, Header, PersonCard, Retention, RoomBaselineCard, SelfReport, SourceControls,
  StateVectors, SystemStatus, Timeline,
} from "@/components/console/panels";
import { PanelBoundary } from "@/components/console/PanelBoundary";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Valence Console — Context-aware affect display" },
      { name: "description", content: "Judge-facing console for Valence: privacy-preserving emotional state inference from room context and body behavior." },
      { property: "og:title", content: "Valence Console" },
      { property: "og:description", content: "Live, privacy-preserving emotional state inference from context, not faces." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Index,
});

function Index() {
  useEffect(() => { boot(); }, []); // a no-op after the root's boot(); keeps the page usable when rendered alone
  const s = useValence();
  const preds = Object.values(s.predictions).sort((a, b) => a.person_id - b.person_id);
  return (
    // On phones the operator panels (source, room baseline) move below the readings so a visitor sees the person
    // card and the emotion map first; the kill switch (Controls) stays near the top. Desktop order is unchanged.
    <div className="flex min-h-screen flex-col">
      <PanelBoundary name="Header"><Header /></PanelBoundary>
      <PanelBoundary name="Welcome"><Welcome /></PanelBoundary>
      <PanelBoundary name="Controls"><Controls /></PanelBoundary>
      <section className="order-last grid gap-5 px-6 pb-6 lg:order-none lg:grid-cols-2 lg:pb-0 lg:pt-5" aria-label="Operator">
        <PanelBoundary name="Source"><SourceControls /></PanelBoundary>
        <PanelBoundary name="Room baseline"><RoomBaselineCard /></PanelBoundary>
      </section>
      <main className="grid gap-5 p-6 xl:grid-cols-[1fr_26rem]">
        <div className="space-y-5">
          <PanelBoundary name="Room view"><CameraView /></PanelBoundary>
          <div className="grid gap-5 2xl:grid-cols-2">
            {preds.length === 0 && <div className="panel text-xl text-muted-foreground">Waiting for predictions…</div>}
            {preds.map((p) => <PanelBoundary key={p.person_id} name={`Person #${p.person_id}`}><PersonCard p={p} /></PanelBoundary>)}
          </div>
          <PanelBoundary name="Emotion map"><AffectMap /></PanelBoundary>
          <PanelBoundary name="Timeline"><Timeline /></PanelBoundary>
        </div>
        <aside className="space-y-5">
          <PanelBoundary name="Self-report"><SelfReport /></PanelBoundary>
          <PanelBoundary name="Governance"><Governance /></PanelBoundary>
          <PanelBoundary name="System status"><SystemStatus /></PanelBoundary>
          <PanelBoundary name="State vector"><StateVectors /></PanelBoundary>
          <PanelBoundary name="Data retention"><Retention /></PanelBoundary>
        </aside>
        <div className="xl:col-span-2"><PanelBoundary name="Evaluation"><Evaluation /></PanelBoundary></div>
      </main>
    </div>
  );
}
