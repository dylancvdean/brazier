/**
 * Playback processor for realtime voice.
 *
 * Decoded frames arrive in bursts over the WebSocket, so they are held in a
 * ring buffer and drained at the audio thread's own pace; an underrun renders
 * silence rather than glitching, and an overrun drops the oldest audio.
 *
 * Loaded by URL rather than inlined as a blob: the renderer's CSP allows
 * scripts from the app origin only.
 */
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.capacity = options.processorOptions.capacity
    this.ring = new Float32Array(this.capacity)
    this.read = 0
    this.write = 0
    this.available = 0
    this.port.onmessage = (event) => {
      if (event.data === 'flush') {
        this.read = 0
        this.write = 0
        this.available = 0
        return
      }
      const chunk = event.data
      for (let index = 0; index < chunk.length; index += 1) {
        this.ring[this.write] = chunk[index]
        this.write = (this.write + 1) % this.capacity
        if (this.available < this.capacity) {
          this.available += 1
        } else {
          this.read = (this.read + 1) % this.capacity
        }
      }
    }
  }

  process(_inputs, outputs) {
    const output = outputs[0][0]
    if (!output) return true
    for (let index = 0; index < output.length; index += 1) {
      if (this.available > 0) {
        output[index] = this.ring[this.read]
        this.read = (this.read + 1) % this.capacity
        this.available -= 1
      } else {
        output[index] = 0
      }
    }
    return true
  }
}

registerProcessor('brazier-playback', PlaybackProcessor)
