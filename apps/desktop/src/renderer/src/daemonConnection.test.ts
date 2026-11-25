import { describe, expect, it } from 'vitest'
import {
  INITIAL_DAEMON_CONNECTION,
  availabilityForEvent,
  daemonConnectionReducer as reduce,
  daemonStatusOf,
  type DaemonConnectionEvent
} from './daemonConnection'

function run(...events: DaemonConnectionEvent[]) {
  return events.reduce(reduce, INITIAL_DAEMON_CONNECTION)
}

describe('daemonConnectionReducer', () => {
  it('goes straight to healthy on first contact', () => {
    expect(run({ type: 'reachable', version: '1.0' })).toEqual({ phase: 'healthy', version: '1.0' })
  })

  it('rehydrates after recovering from offline, then becomes healthy', () => {
    const recovering = run({ type: 'reachable', version: '1.0' }, { type: 'unreachable' }, {
      type: 'reachable'
    })
    expect(recovering).toEqual({ phase: 'rehydrating', version: '1.0' })
    expect(daemonStatusOf(recovering)).toBe('healthy')
    expect(reduce(recovering, { type: 'rehydrated' })).toEqual({ phase: 'healthy', version: '1.0' })
  })

  it('always rehydrates after a manual retry', () => {
    expect(
      run({ type: 'reachable' }, { type: 'retry' }, { type: 'reachable', version: '2.0' })
    ).toEqual({ phase: 'rehydrating', version: '2.0' })
  })

  it('ignores a stray rehydrated event and keeps identity on repeated health', () => {
    const healthy = run({ type: 'reachable', version: '1.0' })
    expect(reduce(healthy, { type: 'rehydrated' })).toBe(healthy)
    expect(reduce(healthy, { type: 'reachable', version: '1.0' })).toBe(healthy)
  })

  it('maps a blocked connection to the blocked request gate', () => {
    const blocked = { type: 'unreachable', blocked: true } as const
    expect(availabilityForEvent(blocked)).toBe('blocked')
    expect(availabilityForEvent({ type: 'unreachable' })).toBe('offline')
    expect(daemonStatusOf(run(blocked))).toBe('offline')
    expect(run(blocked, { type: 'reachable', version: '2.0' })).toEqual({
      phase: 'rehydrating',
      version: '2.0'
    })
  })

  it('goes offline from any phase', () => {
    expect(run({ type: 'retry' }, { type: 'unreachable' }).phase).toBe('offline')
    expect(run({ type: 'unreachable' }, { type: 'reachable' }, { type: 'unreachable' }).phase).toBe(
      'offline'
    )
  })
})
