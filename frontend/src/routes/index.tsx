import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { boot, useValence } from "@/lib/valence";
import {
  CameraView, Controls, Evaluation, Governance, Header, PersonCard, Retention, StateVectors, SystemStatus, Timeline,
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
  useEffect(() => { boot(); }, []);
  const s = useValence();
  const preds = Object.values(s.predictions).sort((a, b) => a.person_id - b.person_id);
  return (
    <div className="min-h-screen">
      <PanelBoundary name="Header"><Header /></PanelBoundary>
      <PanelBoundary name="Controls"><Controls /></PanelBoundary>
      <main className="grid gap-5 p-6 xl:grid-cols-[1fr_26rem]">
        <div className="space-y-5">
          <PanelBoundary name="Room view"><CameraView /></PanelBoundary>
          <div className="grid gap-5 2xl:grid-cols-2">
            {preds.length === 0 && <div className="panel text-xl text-muted-foreground">Waiting for predictions…</div>}
            {preds.map((p) => <PanelBoundary key={p.person_id} name={`Person #${p.person_id}`}><PersonCard p={p} /></PanelBoundary>)}
          </div>
          <PanelBoundary name="Timeline"><Timeline /></PanelBoundary>
        </div>
        <aside className="space-y-5">
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
