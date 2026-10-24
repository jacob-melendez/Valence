/**
 * createPerception(): MediaPipe Tasks Web perception on a MediaStream. Stub for now (interface only): start()
 * resolves, reports "running" with a "stub" degradation and emits nothing. The real pipeline replaces it.
 */
import { Hub } from "./hub";
import {
  emptyCounters, idleStatus, type Obs, type Perception, type PerceptionOptions, type PerceptionStatus,
} from "./types";

export function createPerception(_opts: PerceptionOptions = {}): Perception {
  const obsHub = new Hub<Obs>();
  const statusHub = new Hub<PerceptionStatus>();
  let status = idleStatus(false);
  const setStatus = (patch: Partial<PerceptionStatus>) => {
    status = { ...status, ...patch };
    statusHub.emit(status);
  };
  return {
    async start(stream) {
      if (!stream || !stream.getTracks().length) throw new Error("perception needs a MediaStream with a camera or mic");
      setStatus({
        phase: "running",
        camera: stream.getVideoTracks().length > 0,
        mic: stream.getAudioTracks().length > 0,
        degraded: ["stub: in-browser perception is not implemented yet"],
      });
    },
    stop() {
      if (status.phase === "running") setStatus({ phase: "stopped", camera: false, mic: false });
    },
    onObservation: (cb) => obsHub.on(cb),
    counters: emptyCounters,
    status: () => status,
    onStatus: (cb) => statusHub.on(cb),
  };
}
