import type { LlamaContext } from 'llama.rn';
import { NativeEventEmitter, NativeModules, Platform, type NativeModule } from 'react-native';
import { answerDelta, type ChatTurn, type Runtime } from './engine.ts';
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

type NativeEngine = 'apple';
type OnDevice = NativeModule & {
  availability(engine: NativeEngine): Promise<{ available: boolean; reason: string }>;
  load(engine: NativeEngine, directory: string): Promise<void>;
  generate(engine: NativeEngine, request: string, messages: ChatTurn[]): Promise<string>;
  stop(engine: NativeEngine): void;
  unload(engine: NativeEngine): Promise<void>;
};

let requests = 0;

/** The iOS engines written for this app (plugins/on-device). Absent on Android and in older builds. */
function nativeEngine(engine: NativeEngine) {
  const native = Platform.OS === 'ios' ? NativeModules.BrazierOnDevice as OnDevice | undefined : undefined;
  return {
    available: async () => !!native && (await native.availability(engine)).available,
    load: async (directory: string) => { await native!.load(engine, directory); },
    async generate(messages: ChatTurn[], signal: AbortSignal, onToken: (token: string) => void): Promise<string> {
      const request = `${engine}-${requests += 1}`;
      const events = new NativeEventEmitter(native!).addListener('BrazierOnDeviceText',
        (event: { request: string; text: string }) => { if (event.request === request && !signal.aborted) onToken(event.text); });
      const stop = () => native!.stop(engine);
      signal.addEventListener('abort', stop, { once: true });
      try {
        return await native!.generate(engine, request, messages);
      } finally {
        signal.removeEventListener('abort', stop);
        events.remove();
      }
    },
    unload: async () => { await native?.unload(engine); },
  };
}

/** Apple's system model: iOS 26 with Apple Intelligence turned on. Nothing to download. */
export function appleRuntime(): Runtime {
  const native = nativeEngine('apple');
  return {
    id: 'apple',
    available: native.available,
    builtIn: async () => [{ id: 'apple-intelligence', name: 'Apple Intelligence', engine: 'apple' }],
    load: async () => undefined,
    generate: native.generate,
    unload: native.unload,
  };
}
