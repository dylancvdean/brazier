import { fetch } from 'expo/fetch';
import { catalogueChoice, type CatalogueEntry } from './catalogue.ts';
import { downloadModel } from './device.ts';
import { listRepoFiles } from './huggingface.ts';
import type { InstalledModel } from './models.ts';
import { searchModels, type SearchPage, type SearchRequest } from './search.ts';

/** Hugging Face search from this device. */
export const searchHub = (request: SearchRequest, next?: string | null): Promise<SearchPage> => searchModels(fetch, request, next);

/** Downloads a catalogue model, with the files and sizes its repository lists now. */
export async function downloadCatalogueModel(entry: CatalogueEntry): Promise<InstalledModel> {
  const choice = catalogueChoice(entry, await listRepoFiles(fetch, entry.slug));
  if (!choice) throw new Error(`${entry.name} is no longer in ${entry.slug}. Search for it instead.`);
  return downloadModel(entry.slug, choice);
}
