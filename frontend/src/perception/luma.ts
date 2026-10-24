/**
 * Luminance for the light estimate: OpenCV's `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)` on 8-bit pixels, which
 * valence/perception/video.py feeds to background_stats (lux.py). The browser reads RGBA; Python sees the same pixel
 * as BGR.
 *
 * OpenCV's uchar RGB2Gray is 15-bit fixed point with round-half-up: Y = (9798 R + 19235 G + 3735 B + 2^14) >> 15
 * (RY15 / GY15 / BY15, gray_shift = 15). The often-quoted 14-bit form (4899 R + 9617 G + 1868 B + 2^13) >> 14 is off
 * by one on 0.26% of colours. Checked bit-exact on all 256^3 triples against opencv-python 5.0.0 and 4.12.0 (x86-64,
 * IPP on and off) by perception-fixtures/gen_lux.py.
 */

export const RY15 = 9798;
export const GY15 = 19235;
export const BY15 = 3735;
const GRAY_SHIFT = 15;
const HALF = 1 << (GRAY_SHIFT - 1);

/** Gray level of one 8-bit (r, g, b) pixel, bit-exact with cv2 BGR2GRAY on the BGR pixel (b, g, r). */
export function grayPixel(r: number, g: number, b: number): number {
  return (r * RY15 + g * GY15 + b * BY15 + HALF) >> GRAY_SHIFT;
}

/** Canvas RGBA (row-major width x height, alpha ignored) -> row-major Uint8Array of grayPixel. */
export function rgbaToGray(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number): Uint8Array {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 0 || height < 0
    || rgba.length !== 4 * width * height) {
    throw new RangeError("rgbaToGray expects width x height RGBA pixels");
  }
  const gray = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) gray[i] = grayPixel(rgba[p]!, rgba[p + 1]!, rgba[p + 2]!);
  return gray;
}
