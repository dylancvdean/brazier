/**
 * Microphone capture processor for realtime voice.
 *
 * Batches the audio thread's render quanta into fixed 20 ms frames and posts
 * them to `VoiceStream`, which hands them straight to the Opus encoder.
 *
 * Loaded by URL rather than inlined as a blob: the renderer's CSP allows
 * scripts from the app origin only.
 */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.frameSize = options.processorOptions.frameSize
    this.buffer = new Float32Array(this.frameSize)
    this.filled = 0
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0]
    if (!input) return true
    for (let index = 0; index < input.length; index += 1) {
      this.buffer[this.filled] = input[index]
      this.filled += 1
      if (this.filled === this.frameSize) {
        this.port.postMessage(this.buffer.slice())
        this.filled = 0
      }
    }
    return true
  }
}

registerProcessor('brazier-capture', CaptureProcessor)
