import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { boot, useValence } from "@/lib/valence";
import {
  CameraView, Controls, Evaluation, Governance, Header, PersonCard, Retention, StateVectors, SystemStatus, Timeline,
} from "@/components/console/panels";

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
      <Header />
      <Controls />
      <main className="grid gap-5 p-6 xl:grid-cols-[1fr_26rem]">
        <div className="space-y-5">
          <CameraView />
          <div className="grid gap-5 2xl:grid-cols-2">
            {preds.length === 0 && <div className="panel text-xl text-muted-foreground">Waiting for predictions…</div>}
            {preds.map((p) => <PersonCard key={p.person_id} p={p} />)}
          </div>
          <Timeline />
        </div>
        <aside className="space-y-5">
          <Governance />
          <SystemStatus />
          <StateVectors />
          <Retention />
        </aside>
        <div className="xl:col-span-2"><Evaluation /></div>
      </main>
    </div>
  );
}
