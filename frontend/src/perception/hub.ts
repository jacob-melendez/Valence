/** Tiny listener set: a throwing listener is logged and never stops the others or the perception loop. */
export class Hub<T> {
  private readonly listeners = new Set<(value: T) => void>();

  on(cb: (value: T) => void): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  emit(value: T): void {
    for (const cb of [...this.listeners]) {
      try {
        cb(value);
      } catch (e) {
        console.error("[perception] listener failed", e);
      }
    }
  }
}
