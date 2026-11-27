import type { LlamaContext } from 'llama.rn';
import { NativeEventEmitter, NativeModules, Platform, type NativeModule } from 'react-native';
import { answerDelta, type ChatTurn, type Runtime } from './engine.ts';
import { CONTEXT_TOKENS, contextBytes } from './memory.ts';
import { ImageCancelled, type ImageFile, type ImageProgress, type ImageRequest, type ImageRuntime } from './images.ts';
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
      context = await initLlama({ model: modelPath(model), n_ctx: CONTEXT_TOKENS, n_gpu_layers: 99 });
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

/** The context (KV) cache a GGUF file will need, from its header; null when it cannot be read. */
export async function llamaContextBytes(path: string): Promise<number | null> {
  try {
    const { loadLlamaModelInfo } = await import('llama.rn');
    return contextBytes(await loadLlamaModelInfo(path) as Record<string, unknown>);
  } catch {
    return null;
  }
}

type NativeEngine = 'apple' | 'mlx';
type OnDevice = NativeModule & {
  availability(engine: NativeEngine): Promise<{ available: boolean; reason: string }>;
  load(engine: NativeEngine, directory: string): Promise<void>;
  generate(engine: NativeEngine, request: string, messages: ChatTurn[]): Promise<string>;
  stop(engine: NativeEngine): void;
  unload(engine: NativeEngine): Promise<void>;
  memoryLimit?(): Promise<number | null>;
};

/** Bytes iOS lets this app use before ending it; null on Android, in the Simulator and in older builds. */
export async function appMemoryLimit(): Promise<number | null> {
  const native = Platform.OS === 'ios' ? NativeModules.BrazierOnDevice as OnDevice | undefined : undefined;
  return await native?.memoryLimit?.().catch(() => null) ?? null;
}

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

/** MLX on Apple silicon: iOS 17 on a real device. The model is a directory of files from an MLX repository. */
export function mlxRuntime(modelDirectory: (model: InstalledModel) => string): Runtime {
  const native = nativeEngine('mlx');
  return {
    id: 'mlx',
    available: native.available,
    async load(model) {
      if (!model) throw new Error('MLX needs a downloaded model.');
      await native.load(modelDirectory(model));
    },
    generate: native.generate,
    unload: native.unload,
  };
}

type Sdcpp = NativeModule & {
  load(path: string): Promise<void>;
  generate(request: ImageRequest): Promise<ImageFile>;
  cancel(): void;
  unload(): Promise<void>;
};

/**
 * stable-diffusion.cpp, linked into the app (modules/brazier-sdcpp): single-file image checkpoints, Metal on
 * iPhones with an A14 or newer and the CPU elsewhere. Absent in builds made before the module was added.
 */
export function sdcppRuntime(modelPath: (model: InstalledModel) => string): ImageRuntime {
  const native = NativeModules.BrazierSdcpp as Sdcpp | undefined;
  return {
    id: 'sd.cpp',
    available: async () => !!native,
    load: async (model) => { await native!.load(modelPath(model)); },
    async generate(request, signal, onProgress) {
      const events = new NativeEventEmitter(native!).addListener('BrazierSdcppProgress',
        (event: ImageProgress) => { if (!signal.aborted) onProgress(event); });
      const stop = () => native!.cancel();
      signal.addEventListener('abort', stop, { once: true });
      try {
        return await native!.generate(request);
      } catch (cause) {
        if (signal.aborted || (cause as { code?: string })?.code === 'cancelled') throw new ImageCancelled();
        throw cause;
      } finally {
        signal.removeEventListener('abort', stop);
        events.remove();
      }
    },
    unload: async () => { await native?.unload(); },
  };
}
