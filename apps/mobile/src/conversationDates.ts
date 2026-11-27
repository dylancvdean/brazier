import { dayLabel } from './history.ts';

/** The day a conversation was last written to. A host reports SQLite's UTC `YYYY-MM-DD HH:MM:SS`; this device ISO 8601. */
export function conversationDay(updatedAt: string | undefined, now?: Date): string {
  if (!updatedAt) return '';
  const iso = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(updatedAt) ? `${updatedAt.replace(' ', 'T')}Z` : updatedAt;
  return Number.isNaN(Date.parse(iso)) ? '' : dayLabel(iso, now);
}
