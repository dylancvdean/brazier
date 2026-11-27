// Checks every catalogue entry against Hugging Face: the repository is public and not gated, the file is
// there, the size matches, and a GGUF's architecture is one the bundled llama.cpp can load.
// Run: pnpm --filter @brazier/mobile catalogue:verify
import { readFileSync } from 'node:fs';
import { catalogue, catalogueChoice } from '../src/local/catalogue.ts';
import { listRepoFiles } from '../src/local/huggingface.ts';

let architectures: Set<string> | null = null;
try {
  const source = readFileSync(new URL('../node_modules/llama.rn/cpp/llama-arch.cpp', import.meta.url), 'utf8');
  architectures = new Set([...source.matchAll(/LLM_ARCH_[A-Z0-9_]+,\s*"([^"]+)"/g)].map((match) => match[1]));
} catch {
  console.warn('llama.rn is not installed; skipping the architecture check.');
}

let failures = 0;
const fail = (id: string, message: string) => { failures += 1; console.error(`FAIL ${id}: ${message}`); };

for (const entry of catalogue) {
  try {
    const response = await fetch(`https://huggingface.co/api/models/${entry.slug}?expand[]=gguf&expand[]=gated&expand[]=private`);
    if (!response.ok) { fail(entry.id, `${entry.slug} returned HTTP ${response.status}`); continue; }
    const info = await response.json() as { gated?: unknown; private?: unknown; gguf?: { architecture?: string } };
    if (info.gated || info.private) { fail(entry.id, `${entry.slug} is gated or private`); continue; }
    const choice = catalogueChoice(entry, await listRepoFiles(fetch, entry.slug));
    if (!choice) { fail(entry.id, `${entry.slug} has no ${entry.entry || 'MLX model'}`); continue; }
    if (choice.size !== entry.size) { fail(entry.id, `size is ${choice.size}, catalogue says ${entry.size}`); continue; }
    const architecture = info.gguf?.architecture;
    if (entry.runtime === 'llama.cpp' && architectures && (!architecture || !architectures.has(architecture))) {
      fail(entry.id, `architecture ${architecture ?? 'unknown'} is not in the bundled llama.cpp`); continue;
    }
    console.log(`ok   ${entry.id}  ${entry.slug}  ${entry.entry || '(repository)'}  ${choice.size}${architecture ? `  ${architecture}` : ''}`);
  } catch (cause) {
    fail(entry.id, cause instanceof Error ? cause.message : String(cause));
  }
}
if (failures) { console.error(`${failures} of ${catalogue.length} entries failed.`); process.exit(1); }
console.log(`All ${catalogue.length} catalogue entries exist on Hugging Face.`);
