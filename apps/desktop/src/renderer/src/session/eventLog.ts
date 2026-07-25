/**
 * Event bus for the session coordinator.
 *
 * Streaming voice and a long-running agent must not block each other, so every
 * hand-off between them is an event. Delivery is idempotent: a repeated final
 * transcript or final agent response carries the same dedupe key and is dropped
 * before any listener sees it, which is what keeps one utterance from starting
 * two agent runs or being spoken twice.
 */

import type { SessionEvent, SessionEventType } from './types'

export type SessionEventListener = (event: SessionEvent) => void

/** How many dedupe keys to remember. Well past any plausible replay window. */
const DEDUPE_CAPACITY = 512

export class SessionEventLog {
  private readonly listeners = new Set<SessionEventListener>()
  private readonly seen = new Set<string>()
  private readonly seenOrder: string[] = []
  private history: SessionEvent[] = []
  private duplicates = 0

  subscribe(listener: SessionEventListener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  /**
   * Publish an event. `dedupeKey` identifies the underlying fact rather than
   * the delivery — an utterance id, or `correlationId:AGENT_RESPONSE_FINAL` —
   * so a resent event is recognized even with a fresh event id.
   *
   * Returns false when the event was a duplicate and was not published.
   */
  emit(event: SessionEvent, dedupeKey?: string): boolean {
    const key = dedupeKey ?? event.eventId
    if (this.seen.has(key)) {
      this.duplicates += 1
      return false
    }
    this.remember(key)
    this.history.push(event)
    if (this.history.length > DEDUPE_CAPACITY) {
      this.history = this.history.slice(-DEDUPE_CAPACITY)
    }
    for (const listener of [...this.listeners]) listener(event)
    return true
  }

  /** Whether a fact with this key has already been published. */
  hasSeen(key: string): boolean {
    return this.seen.has(key)
  }

  /** Record a key without publishing, for facts handled outside the bus. */
  remember(key: string): void {
    this.seen.add(key)
    this.seenOrder.push(key)
    if (this.seenOrder.length > DEDUPE_CAPACITY) {
      const evicted = this.seenOrder.shift()
      if (evicted !== undefined) this.seen.delete(evicted)
    }
  }

  /** Count of events dropped as duplicates, for the metrics report. */
  duplicateCount(): number {
    return this.duplicates
  }

  recent(limit = 50): SessionEvent[] {
    return this.history.slice(-limit)
  }

  eventsOfType(type: SessionEventType): SessionEvent[] {
    return this.history.filter((event) => event.type === type)
  }
}
