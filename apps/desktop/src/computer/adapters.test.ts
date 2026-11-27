import { describe, expect, it } from 'vitest'

import {
  buildComputerUseTool,
  normalizeComputerActionCoordinates,
  parseComputerCompletion,
  parsePromptedModelOutput,
  parseUiTarsOutput,
  resolveComputerAdapter
} from './adapters'

describe('resolveComputerAdapter', () => {
  it('selects fara-xml for Fara1.5 models', () => {
    expect(
      resolveComputerAdapter('gguf:bartowski/Fara1.5-4B-GGUF/Fara1.5-4B-Q4_K_M.gguf')
    ).toMatchObject({
      dialect: 'fara-xml',
      coordinateSpace: 'norm1000'
    })
  })

  it('selects ui-tars for ByteDance UI-TARS models', () => {
    expect(resolveComputerAdapter('ByteDance-Seed/UI-TARS-1.5-7B')).toMatchObject({
      dialect: 'ui-tars',
      coordinateSpace: 'norm1000'
    })
  })

  it('selects openai-tools and norm1000 for Qwen3-VL, Gemma 4, and remote tool VLMs', () => {
    expect(
      resolveComputerAdapter('Qwen/Qwen3-VL-8B-Instruct', {
        id: 'Qwen/Qwen3-VL-8B-Instruct',
        capabilities: { tools: true }
      })
    ).toMatchObject({ dialect: 'openai-tools', coordinateSpace: 'norm1000' })

    expect(
      resolveComputerAdapter('google/gemma-4-27b-it', {
        id: 'google/gemma-4-27b-it',
        capabilities: { tools: true }
      })
    ).toMatchObject({ dialect: 'openai-tools', coordinateSpace: 'norm1000' })

    expect(
      resolveComputerAdapter('remote:openai:gpt-4.1', {
        id: 'remote:openai:gpt-4.1',
        engine: 'remote',
        capabilities: { tools: true }
      })
    ).toMatchObject({ dialect: 'openai-tools', coordinateSpace: 'norm1000' })
  })

  it('selects pixel coordinates for Qwen2.5-VL, Llama 4, and Pixtral', () => {
    expect(
      resolveComputerAdapter('Qwen/Qwen2.5-VL-7B-Instruct', {
        id: 'Qwen/Qwen2.5-VL-7B-Instruct',
        capabilities: { tools: true }
      })
    ).toMatchObject({ dialect: 'openai-tools', coordinateSpace: 'pixels' })

    expect(resolveComputerAdapter('meta-llama/Llama-4-Scout-17B-16E-Instruct')).toMatchObject({
      dialect: 'openai-tools',
      coordinateSpace: 'pixels'
    })

    expect(resolveComputerAdapter('mistralai/Pixtral-12B-2409')).toMatchObject({
      dialect: 'openai-tools',
      coordinateSpace: 'pixels'
    })
  })

  it('falls back to prompted-json for vision-only models without native tool calling', () => {
    expect(
      resolveComputerAdapter('mlx-vlm:mlx-community/Qwen2.5-VL-7B-Instruct-4bit', {
        id: 'mlx-vlm:mlx-community/Qwen2.5-VL-7B-Instruct-4bit',
        engine: 'mlx-vlm',
        capabilities: { tools: false }
      })
    ).toMatchObject({ dialect: 'prompted-json', coordinateSpace: 'pixels' })
  })
})

describe('buildComputerUseTool', () => {
  it('exposes the full computer_use schema and restricts browser-only actions on desktop', () => {
    const browserTool = buildComputerUseTool('browser', 'norm1000', { width: 1280, height: 800 })
    const props = browserTool.function.parameters.properties as Record<string, { enum?: string[] }>
    expect(Object.keys(props)).toEqual([
      'action',
      'coordinate',
      'start_coordinate',
      'text',
      'keys',
      'delta_x',
      'delta_y',
      'url',
      'query',
      'fact',
      'question',
      'response',
      'milliseconds'
    ])
    expect(props.action.enum).toContain('visit_url')
    expect(props.action.enum).toContain('web_search')

    const desktopTool = buildComputerUseTool('desktop', 'pixels', { width: 1920, height: 1080 })
    const desktopProps = desktopTool.function.parameters.properties as Record<
      string,
      { enum?: string[] }
    >
    expect(desktopProps.action.enum).not.toContain('visit_url')
    expect(desktopProps.action.enum).not.toContain('web_search')
    expect(desktopTool.function.description).toContain('1920')
  })
})

describe('universal dialect parsing and coordinate normalization', () => {
  it('recovers Fara <tool_call> even when a local runtime places it inside reasoningText', () => {
    const parsed = parseComputerCompletion(
      {
        responseText: '',
        reasoningText:
          'I see the login button at [640, 320].\n<tool_call>{"name":"computer_use","arguments":{"action":"left_click","coordinate":[640,320]}}</tool_call>'
      },
      { dialect: 'fara-xml', coordinateSpace: 'norm1000' }
    )
    expect(parsed.thought).toBe('I see the login button at [640, 320].')
    expect(parsed.actions).toEqual([{ type: 'left_click', x: 640, y: 320 }])
    expect(parsed.assistantContent).toContain('<tool_call>')
  })

  it('parses native OpenAI clientToolCalls and scales pixel coordinates by viewport', () => {
    const parsed = parseComputerCompletion(
      {
        responseText: '',
        reasoningText: 'Clicking the center of the 1920x1080 viewport.',
        clientToolCalls: [
          {
            id: 'call_1',
            name: 'computer_use',
            arguments: JSON.stringify({
              action: 'left_click',
              coordinate: [960, 540]
            })
          }
        ]
      },
      {
        dialect: 'openai-tools',
        coordinateSpace: 'pixels',
        viewport: { width: 1920, height: 1080 }
      }
    )
    expect(parsed.thought).toContain('1920x1080')
    expect(parsed.actions).toEqual([{ type: 'left_click', x: 500, y: 500 }])
    expect(parsed.assistantContent).toContain('"action":"left_click"')
  })

  it('scales norm1 [0..1] unit floats into 0..1000', () => {
    expect(
      normalizeComputerActionCoordinates(
        { type: 'left_click_drag', start_x: 0.1, start_y: 0.2, end_x: 0.8, end_y: 0.9 },
        'norm1'
      )
    ).toEqual({
      type: 'left_click_drag',
      start_x: 100,
      start_y: 200,
      end_x: 800,
      end_y: 900
    })
  })

  it('parses ByteDance UI-TARS actions', () => {
    expect(
      parseUiTarsOutput("Thought: Open the folder.\nAction: left_double(start_box='(250,400)')")
    ).toMatchObject({
      thought: 'Open the folder.',
      actions: [{ type: 'double_click', x: 250, y: 400 }]
    })

    expect(parseUiTarsOutput("Action: right_single(start_box='(100,200)')").actions).toEqual([
      { type: 'right_click', x: 100, y: 200 }
    ])

    expect(
      parseUiTarsOutput("Action: drag(start_box='(100,150)', end_box='(600,650)')").actions
    ).toEqual([
      { type: 'left_click_drag', start_x: 100, start_y: 150, end_x: 600, end_y: 650 }
    ])

    expect(parseUiTarsOutput("Action: hotkey(key='ctrl c')").actions).toEqual([
      { type: 'keypress', keys: ['ctrl', 'c'] }
    ])

    expect(parseUiTarsOutput("Action: type(content='hello\\n')").actions).toEqual([
      { type: 'type', text: 'hello\n' }
    ])

    expect(
      parseUiTarsOutput("Action: scroll(start_box='(500,500)', direction='down')").actions
    ).toEqual([{ type: 'scroll', x: 500, y: 500, delta_x: 0, delta_y: 400 }])

    expect(parseUiTarsOutput('Action: wait()').actions).toEqual([
      { type: 'wait', milliseconds: 1000 }
    ])

    expect(parseUiTarsOutput("Action: finished(content='Order found.')").actions).toEqual([
      { type: 'terminate', response: 'Order found.' }
    ])

    expect(parseUiTarsOutput("Action: call_user(content='2FA code needed')").actions).toEqual([
      { type: 'ask_user', question: '2FA code needed' }
    ])
  })

  it('tolerantly parses XML <function=computer_use>, Markdown json fences, and prose JSON', () => {
    const xmlFn = parsePromptedModelOutput(
      'I will click submit.\n<function=computer_use><parameter=action>left_click</parameter><parameter=coordinate>[710, 820]</parameter></function>'
    )
    expect(xmlFn.thought).toBe('I will click submit.')
    expect(xmlFn.actions).toEqual([{ type: 'left_click', x: 710, y: 820 }])

    const fenced = parsePromptedModelOutput(
      'Next action:\n```json\n{"name":"computer_use","arguments":{"action":"key","keys":["Ctrl+L"]}}\n```'
    )
    expect(fenced.actions).toEqual([{ type: 'keypress', keys: ['Ctrl', 'L'] }])

    const prose = parsePromptedModelOutput(
      'Let us navigate to the docs page now: {"name":"computer_use","arguments":{"action":"visit_url","url":"https://docs.example.com"}} thanks!'
    )
    expect(prose.thought).toContain('navigate to the docs page')
    expect(prose.actions).toEqual([{ type: 'visit_url', url: 'https://docs.example.com' }])
  })
})

describe('parsePromptedModelOutput', () => {
  it('finds an action that follows an unbalanced brace in prose', () => {
    const parsed = parsePromptedModelOutput(
      'The form shows a stray { character. {"action":"left_click","coordinate":[10,20]}'
    )
    expect(parsed.actions).toEqual([{ type: 'left_click', x: 10, y: 20 }])
  })
})
