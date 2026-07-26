export type Role = 'system' | 'user' | 'assistant' | 'tool'

export type Conversation = {
  id: string
  title: string
  created_at: string
  updated_at: string
  /** Agent session this conversation's turns go to, when one is bound. */
  agent_session_id?: string | null
  /** Compact summary a fresh voice session is seeded with. */
  summary?: string | null
  summary_updated_at?: string | null
}

export type ContentPart =
  | { type: 'text'; text: string }
  | {
      type: 'brazier_blob'
      brazier_blob: { sha256: string; mime_type: string; name: string }
    }
  | { type: 'image_url'; image_url: { url: string } }
  | { type: 'input_audio'; input_audio: { data: string; format: string } }
  | { type: 'input_video'; video_url: { url: string } }

export type Message = {
  id: string
  conversation_id: string
  parent_id: string | null
  role: Role
  content: string | ContentPart[]
  model: string | null
  tool_calls?: unknown[] | null
  tool_call_id?: string | null
  /** Which surface produced it: `user_voice`, `assistant_agent`, … */
  source?: string | null
  /** Ties a user turn to its authoritative answer and any voice experiment. */
  correlation_id?: string | null
  /** `partial`, `final`, `cancelled`, `superseded`, `failed`. */
  status?: string | null
  metadata?: Record<string, unknown> | null
  created_at: string
}

export type Attachment = {
  id: string
  name: string
  type: string
  sha256: string
}

export type HubModel = {
  id: string
  author: string
  downloads: number
  likes: number
  last_modified: string | null
  tags: string[]
  gated: boolean
  score: number
  preferred_quantizer: boolean
}
