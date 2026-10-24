"""Fake camera and microphone files for the end-to-end perception test (frontend/e2e/run.mjs).

Chromium plays these through --use-file-for-fake-video-capture (Y4M) and --use-file-for-fake-audio-capture (WAV),
looping. Sources are freely licensed (frontend/e2e/README.md); nobody is recorded.

    .venv/bin/python frontend/e2e/make_media.py [out_dir]      (default frontend/e2e/.media)

Scenes (1280x720, 10 fps, 3 s loops), all built from media/portrait.jpg:
- one.y4m: the portrait at full frame height in a plain room, drifting a few pixels (some motion);
- two.y4m: two copies side by side (the right one mirrored), for tracking and face-to-person assignment;
- roll_cw.y4m / roll_ccw.y4m: the one-person scene rotated 15 degrees clockwise / counterclockwise, to check the sign
  of head roll.
Audio: speech.wav (48 kHz mono 16-bit) from media/speech.ogg, decoded with the ffmpeg bundled in imageio-ffmpeg.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
W, H, FPS, SECONDS = 1280, 720, 10, 3


def room(rng: np.random.Generator) -> Image.Image:
    """A plain wall: vertical gradient plus mild sensor-like noise."""
    y = np.linspace(170, 120, H)[:, None, None]
    base = np.broadcast_to(y, (H, W, 3)) * np.array([1.0, 0.97, 0.92])
    noisy = base + rng.normal(0, 3, (H, W, 3))
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def scene(kind: str, frame: int, portrait: Image.Image, rng: np.random.Generator) -> Image.Image:
    canvas = room(rng)
    if kind == "two":
        p = portrait.resize((600, 600), Image.LANCZOS)
        canvas.paste(p, (20, H - 600))
        canvas.paste(p.transpose(Image.FLIP_LEFT_RIGHT), (W - 620, H - 600))
        return canvas
    dx = int(round(8 * math.sin(2 * math.pi * frame / (FPS * SECONDS))))
    p = portrait.resize((H, H), Image.LANCZOS)
    canvas.paste(p, ((W - H) // 2 + dx, 0))
    if kind in ("roll_cw", "roll_ccw"):
        angle = -15 if kind == "roll_cw" else 15  # PIL rotates counterclockwise for positive angles
        canvas = canvas.rotate(angle, resample=Image.BICUBIC, fillcolor=(140, 136, 128))
    return canvas


def rgb_to_i420(img: Image.Image) -> bytes:
    """BT.601 limited-range I420, what Chromium's Y4M reader expects."""
    a = np.asarray(img, dtype=np.float64)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    y = 16 + 0.257 * r + 0.504 * g + 0.098 * b
    u = 128 - 0.148 * r - 0.291 * g + 0.439 * b
    v = 128 + 0.439 * r - 0.368 * g - 0.071 * b
    sub = lambda c: c.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))  # noqa: E731
    planes = [y, sub(u), sub(v)]
    return b"".join(np.clip(np.round(p), 0, 255).astype(np.uint8).tobytes() for p in planes)


def write_y4m(path: Path, kind: str, portrait: Image.Image) -> None:
    rng = np.random.default_rng(7)
    with path.open("wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n".encode())
        for i in range(FPS * SECONDS):
            f.write(b"FRAME\n" + rgb_to_i420(scene(kind, i, portrait, rng)))


def write_wav(path: Path) -> None:
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(HERE / "media" / "speech.ogg"),
                    "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", str(path)], check=True)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / ".media"
    out.mkdir(parents=True, exist_ok=True)
    portrait = Image.open(HERE / "media" / "portrait.jpg").convert("RGB")
    for kind in ("one", "two", "roll_cw", "roll_ccw"):
        write_y4m(out / f"{kind}.y4m", kind, portrait)
        scene(kind, 0, portrait, np.random.default_rng(7)).save(out / f"{kind}.png")  # preview of the first frame
    write_wav(out / "speech.wav")
    print(f"media in {out}: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
