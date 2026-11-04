// Valence Web audio tap (AudioWorklet). Copies channel 0 of the mic input into ~100 ms Float32Array chunks and
// transfers each to the page, where perception analyses it and zero-fills it. Nothing is stored or sent anywhere.
class PcmTap extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.size = Math.max(128, Math.round(opts.chunkFrames || sampleRate / 10));
    this.buf = new Float32Array(this.size);
    this.n = 0;
    this.running = true;
    this.port.onmessage = (e) => {
      if (e.data === "stop") {
        this.running = false;
        this.buf.fill(0);
      } else if (e.data === "flush") {
        // The context was suspended mid-chunk: drop the partial chunk instead of joining it to later audio.
        this.buf.fill(0);
        this.n = 0;
      }
    };
  }

  process(inputs) {
    if (!this.running) return false;
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    let i = 0;
    while (i < channel.length) {
      const take = Math.min(this.size - this.n, channel.length - i);
      this.buf.set(channel.subarray(i, i + take), this.n);
      this.n += take;
      i += take;
      if (this.n === this.size) {
        this.port.postMessage(this.buf, [this.buf.buffer]);
        this.buf = new Float32Array(this.size);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor("valence-pcm-tap", PcmTap);
