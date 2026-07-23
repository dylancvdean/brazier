export type Role = 'system' | 'user' | 'assistant' | 'tool'

export type Conversation = {
  id: string
  title: string
  created_at: string
  updated_at: string
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
