/**
 * getUserMedia with the constraints perception expects. The browser's speech processing (echo cancellation, noise
 * suppression, automatic gain) is turned off: it would pump the room level and erase laughter, and the Python edge
 * reads the raw mic too. Video asks for 1280x720 like the Python capture.
 */

export const VIDEO_CONSTRAINTS: MediaTrackConstraints = {
  width: { ideal: 1280 },
  height: { ideal: 720 },
  frameRate: { ideal: 30 },
  facingMode: "user",
};

export const AUDIO_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1,
};

export interface OpenedMedia {
  /** null when neither the camera nor the mic could be opened. */
  stream: MediaStream | null;
  camera: boolean;
  mic: boolean;
  /** One line per device that failed, e.g. "camera: NotAllowedError: Permission denied". */
  errors: string[];
}

const describe = (what: string, e: unknown): string =>
  `${what}: ${e instanceof Error ? `${e.name}: ${e.message}` : String(e)}`;

/**
 * Open the camera and the mic. Asks for both at once (one permission prompt), then falls back to each alone, so a
 * denied or missing mic still leaves the camera running and vice versa.
 */
export async function openMediaStream(
  want: { video?: boolean; audio?: boolean } = {},
  media: Pick<MediaDevices, "getUserMedia"> | undefined = globalThis.navigator?.mediaDevices,
): Promise<OpenedMedia> {
  const video = want.video ?? true;
  const audio = want.audio ?? true;
  const errors: string[] = [];
  if (!media?.getUserMedia) {
    return { stream: null, camera: false, mic: false, errors: ["getUserMedia is not available (needs HTTPS)"] };
  }
  if (video && audio) {
    try {
      const stream = await media.getUserMedia({ video: VIDEO_CONSTRAINTS, audio: AUDIO_CONSTRAINTS });
      return { stream, camera: stream.getVideoTracks().length > 0, mic: stream.getAudioTracks().length > 0, errors };
    } catch (e) {
      errors.push(describe("camera + mic", e));
    }
  }
  const tracks: MediaStreamTrack[] = [];
  let camera = false;
  let mic = false;
  if (video) {
    try {
      const s = await media.getUserMedia({ video: VIDEO_CONSTRAINTS });
      tracks.push(...s.getVideoTracks());
      camera = s.getVideoTracks().length > 0;
    } catch (e) {
      errors.push(describe("camera", e));
    }
  }
  if (audio) {
    try {
      const s = await media.getUserMedia({ audio: AUDIO_CONSTRAINTS });
      tracks.push(...s.getAudioTracks());
      mic = s.getAudioTracks().length > 0;
    } catch (e) {
      errors.push(describe("mic", e));
    }
  }
  if (!tracks.length) return { stream: null, camera, mic, errors };
  return { stream: new MediaStream(tracks), camera, mic, errors };
}
