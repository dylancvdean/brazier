import type { LlamaContext } from 'llama.rn';
import { answerDelta, type Runtime } from './engine.ts';
import type { InstalledModel } from './models.ts';

/** llama.cpp through llama.rn: GGUF files, Metal on iOS. */
export function llamaRuntime(modelPath: (model: InstalledModel) => string): Runtime {
  let context: LlamaContext | null = null;
  return {
    id: 'llama.cpp',
    available: async () => true,
    async load(model) {
      if (!model) throw new Error('llama.cpp needs a downloaded model.');
      const { initLlama } = await import('llama.rn');
      // llama.rn falls back to the CPU where the GPU cannot be used, such as the iOS Simulator.
      context = await initLlama({ model: modelPath(model), n_ctx: 4096, n_gpu_layers: 99 });
    },
    async generate(messages, signal, onToken) {
      const current = context;
      if (!current) throw new Error('The model is not loaded.');
      const stop = () => { void current.stopCompletion().catch(() => undefined); };
      signal.addEventListener('abort', stop, { once: true });
      let shown = '';
      try {
        const result = await current.completion({ messages, jinja: true, reasoning_format: 'auto' }, (data) => {
          const added = answerDelta(shown, data);
          if (!added || signal.aborted) return;
          shown += added;
          onToken(added);
        });
        return result.content || shown || result.text;
      } finally {
        signal.removeEventListener('abort', stop);
      }
    },
    async unload() {
      const current = context;
      context = null;
      await current?.release();
    },
  };
}
