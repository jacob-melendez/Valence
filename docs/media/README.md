# Media

| File | What | How it was made |
|---|---|---|
| `valence-web-tour.mp4` | 75 s screen recording of the hosted Valence Web page on the camera-free tour (30x): welcome banner, a person card filling in, the emotion map's trail with the scripted state as a dashed ring, the governance panel's verified audit chain | `cd frontend && npm run build && node scripts/record-demo.mjs --out ../docs/media/valence-web-tour.webm`, then `ffmpeg -i ... -c:v libx264 -crf 28 -pix_fmt yuv420p -movflags +faststart` |

The tour is a **scripted, simulated** session (no camera, no real person); the page's badge says so, and so must any
caption. It is a backup for the demo; the live-camera part has to be recorded on a real machine.
