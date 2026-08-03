import { describe, expect, it } from 'vitest'

import {
  EMPTY_OMP_SIDECAR,
  commandSuggestion,
  frameDetail,
  ompSidecarReducer
} from './ompSidecar'

describe('ompSidecarReducer', () => {
  it('renders command output as bounded blocks', () => {
    let state = EMPTY_OMP_SIDECAR
    for (let index = 0; index < 20; index++) {
      state = ompSidecarReducer(state, {
        type: 'frame',
        frame: { type: 'command_output', text: `output ${index}` }
      })
    }
    expect(state.commandOutputs).toHaveLength(12)
    expect(state.commandOutputs.at(-1)).toMatchObject({ text: 'output 19' })
  })

  it('drops empty command output', () => {
    const state = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: { type: 'command_output', text: '' }
    })
    expect(state.commandOutputs).toHaveLength(0)
  })

  it('keeps the live slash-command list from available_commands_update', () => {
    const state = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: {
        type: 'available_commands_update',
        commands: [
          { name: '/review', description: 'Review the change', source: 'builtin' },
          { name: 'vibe', description: 'Enter vibe mode', source: 'builtin' }
        ]
      }
    })
    expect(state.commands).toEqual([
      { value: '/review', description: 'Review the change' },
      { value: '/vibe', description: 'Enter vibe mode' }
    ])
  })

  it('replaces the command list with each full available_commands_update snapshot', () => {
    const first = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: {
        type: 'available_commands_update',
        commands: [{ name: '/model', description: 'Pick a model', source: 'builtin' }]
      }
    })
    const second = ompSidecarReducer(first, {
      type: 'frame',
      frame: {
        type: 'available_commands_update',
        commands: [
          { name: 'model', description: 'Pick a model', source: 'builtin' },
          { name: '/fresh', description: 'Reset the stream', source: 'builtin' }
        ]
      }
    })
    expect(second.commands).toEqual([
      { value: '/model', description: 'Pick a model' },
      { value: '/fresh', description: 'Reset the stream' }
    ])
    // A removed command disappears from the next snapshot.
    const third = ompSidecarReducer(second, {
      type: 'frame',
      frame: { type: 'available_commands_update', commands: [{ name: '/fresh', source: 'builtin' }] }
    })
    expect(third.commands).toEqual([{ value: '/fresh', description: 'OMP command' }])
  })

  it('tracks session metadata from session_info_update and config_update', () => {
    let state = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: { type: 'session_info_update', title: 'Task title', sessionId: 'sess-1' }
    })
    state = ompSidecarReducer(state, {
      type: 'frame',
      frame: { type: 'config_update', model: { id: 'gguf:model.gguf' }, thinkingLevel: 'high' }
    })
    expect(state.session).toMatchObject({
      title: 'Task title',
      sessionId: 'sess-1',
      modelName: 'gguf:model.gguf',
      thinkingLevel: 'high'
    })
  })

  it('routes unknown frame types into the bounded generic record', () => {
    let state = EMPTY_OMP_SIDECAR
    for (let index = 0; index < 50; index++) {
      state = ompSidecarReducer(state, {
        type: 'frame',
        frame: { type: 'some_future_frame', detail: String(index) }
      })
    }
    expect(state.recentFrames).toHaveLength(30)
    expect(state.recentFrames.at(-1)).toMatchObject({ type: 'some_future_frame' })
  })

  it('ignores frames the shared transcript path already renders', () => {
    const state = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: { type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: 'x' } }
    })
    expect(state.recentFrames).toHaveLength(0)
    expect(state.commandOutputs).toHaveLength(0)
  })

  it('resets on the reset action', () => {
    const seeded = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: { type: 'command_output', text: 'hello' }
    })
    const reset = ompSidecarReducer(seeded, { type: 'reset' })
    expect(reset).toEqual(EMPTY_OMP_SIDECAR)
  })

  it('folds get_state snapshots into session metadata', () => {
    const state = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: {
        type: 'response',
        command: 'get_state',
        success: true,
        data: {
          sessionId: 'sess-1',
          sessionName: 'OMP task',
          model: { provider: 'brazier', id: 'gguf:model.gguf' },
          thinkingLevel: 'high',
          fastModeEnabled: true,
          fastModeActive: true,
          autoCompactionEnabled: false,
          isStreaming: false,
          isCompacting: false,
          tokensPerSecond: 12,
          contextUsage: { tokens: 40000, contextWindow: 200000, percent: 20 },
          todoPhases: [
            { id: 'phase-1', name: 'T', tasks: [{ id: 't1', content: 'Do it', status: 'in_progress' }] }
          ]
        }
      }
    })
    expect(state.session).toMatchObject({
      title: 'OMP task',
      sessionId: 'sess-1',
      modelId: 'gguf:model.gguf',
      modelName: 'gguf:model.gguf',
      thinkingLevel: 'high',
      fastModeEnabled: true,
      fastModeActive: true,
      autoCompactionEnabled: false,
      tokensPerSecond: 12,
      contextUsage: { tokens: 40000, contextWindow: 200000, percent: 20 },
      todoPhases: [{ id: 'phase-1', name: 'T', tasks: [{ id: 't1', content: 'Do it', status: 'in_progress' }] }]
    })
  })

  it('updates fast mode and thinking level from command responses', () => {
    let state = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: { type: 'response', command: 'set_fast_mode', success: true, data: { enabled: true, active: true } }
    })
    expect(state.session).toMatchObject({ fastModeEnabled: true, fastModeActive: true })
    state = ompSidecarReducer(state, {
      type: 'frame',
      frame: { type: 'response', command: 'cycle_thinking_level', success: true, data: { level: 'max' } }
    })
    expect(state.session).toMatchObject({ thinkingLevel: 'max' })
  })

  it('tracks thinking level changes from events', () => {
    const state = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: { type: 'thinking_level_changed', thinkingLevel: 'low' }
    })
    expect(state.session).toMatchObject({ thinkingLevel: 'low' })
  })

  it('merges todo_reminder items into existing phases by id', () => {
    const seeded = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: {
        type: 'response',
        command: 'get_state',
        success: true,
        data: {
          todoPhases: [
            { id: 'phase-1', name: 'T', tasks: [{ id: 't1', content: 'Do it', status: 'pending' }] }
          ]
        }
      }
    })
    const state = ompSidecarReducer(seeded, {
      type: 'frame',
      frame: {
        type: 'todo_reminder',
        todos: [{ id: 't1', content: 'Do it', status: 'in_progress' }],
        attempt: 1,
        maxAttempts: 2
      }
    })
    expect(state.session?.todoPhases?.[0]?.tasks[0]).toMatchObject({ id: 't1', status: 'in_progress' })
  })

  it('clears todos on todo_auto_clear', () => {
    const seeded = ompSidecarReducer(EMPTY_OMP_SIDECAR, {
      type: 'frame',
      frame: {
        type: 'response',
        command: 'get_state',
        success: true,
        data: { todoPhases: [{ id: 'p', name: 'T', tasks: [] }] }
      }
    })
    const cleared = ompSidecarReducer(seeded, { type: 'frame', frame: { type: 'todo_auto_clear' } })
    expect(cleared.session?.todoPhases).toEqual([])
  })
})

describe('frameDetail', () => {
  it('summarizes notices and thinking changes', () => {
    expect(frameDetail({ type: 'notice', message: 'Disk is low' })).toBe('Disk is low')
    expect(frameDetail({ type: 'thinking_level_changed', thinkingLevel: 'high' })).toBe('thinking → high')
  })

  it('falls back to the raw frame type for anything new', () => {
    expect(frameDetail({ type: 'brand_new_event' })).toBe('brand_new_event')
  })
})

describe('commandSuggestion', () => {
  it('adds a leading slash to bare command names', () => {
    expect(commandSuggestion({ name: 'advisor', description: 'Run the advisor', source: 'builtin' })).toEqual({
      value: '/advisor',
      description: 'Run the advisor'
    })
  })
})
