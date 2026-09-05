import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { MessageMedia } from './MessageMedia'

describe('attachment display', () => {
  it('offers a save action for documents without a permanent loading placeholder', () => {
    const html = renderToStaticMarkup(<MessageMedia blobs={[
      { sha256: 'pdf', mime_type: 'application/pdf', original_name: 'Report.pdf' }
    ]} />)
    expect(html).toContain('Report.pdf')
    expect(html).toContain('Save this file to open it.')
    expect(html).not.toContain('Loading')
  })
})
