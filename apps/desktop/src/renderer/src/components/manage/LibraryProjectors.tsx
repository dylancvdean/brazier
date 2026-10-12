import { Download, FolderOpen, LoaderCircle, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import {
  attachProjector,
  deleteProjector,
  formatBytes,
  listDownloadJobs,
  listHubFiles,
  queueModelDownload,
  type HubFile,
  type LocalModel
} from '../../api'

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function basename(path: string): string {
  return path.split('/').at(-1) ?? path
}

export function LibraryProjectors(props: {
  model: LocalModel
  /** Other quants in the same folder, including this model. */
  sharedWith: number
  allowLocalFile: boolean
  onError: (message: string | null) => void
  onChanged: () => Promise<void>
}): React.JSX.Element {
  const projectors = props.model.projectors ?? []
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [options, setOptions] = useState<HubFile[] | null>(null)
  const [queued, setQueued] = useState<string | null>(null)
  const [removing, setRemoving] = useState<string | null>(null)
  const [confirming, setConfirming] = useState<string | null>(null)
  const [attaching, setAttaching] = useState(false)

  useEffect(() => {
    if (!queued) return
    const wanted = basename(queued).toLowerCase()
    if (projectors.some((projector) => projector.name.toLowerCase() === wanted)) {
      setQueued(null)
      setOpen(false)
    }
  }, [projectors, queued])

  const onErrorRef = useRef(props.onError)
  const onChangedRef = useRef(props.onChanged)
  onErrorRef.current = props.onError
  onChangedRef.current = props.onChanged

  useEffect(() => {
    if (!queued) return
    let cancelled = false
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const jobs = await listDownloadJobs()
          if (cancelled) return
          const wanted = basename(queued)
          const job = jobs.find(
            (entry) => entry.filename === queued || basename(entry.filename) === wanted
          )
          if (!job) return
          if (job.status === 'failed' || job.status === 'cancelled') {
            setQueued(null)
            onErrorRef.current(job.error || `Projector download ${job.status}.`)
            return
          }
          if (job.status === 'completed') {
            await onChangedRef.current()
          }
        } catch {
          // The tray still shows the job if this poll misses.
        }
      })()
    }, 3000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [queued])

  async function loadOptions(): Promise<void> {
    if (open) {
      setOpen(false)
      return
    }
    setOpen(true)
    if (!props.model.repo_id) {
      setOptions([])
      return
    }
    if (options) return
    setLoading(true)
    props.onError(null)
    try {
      const response = await listHubFiles(props.model.repo_id)
      const installed = new Set(projectors.map((projector) => projector.name.toLowerCase()))
      setOptions(
        response.data.filter((file) => {
          const name = basename(file.path).toLowerCase()
          return name.endsWith('.gguf') && name.includes('mmproj') && !installed.has(name)
        })
      )
    } catch (cause) {
      props.onError(errorText(cause))
      setOptions([])
    } finally {
      setLoading(false)
    }
  }

  async function download(file: HubFile): Promise<void> {
    if (!props.model.repo_id) return
    setQueued(file.path)
    props.onError(null)
    try {
      await queueModelDownload(props.model.repo_id, file.path, 'main', 'llama.cpp', props.model.id)
    } catch (cause) {
      setQueued(null)
      props.onError(errorText(cause))
    }
  }

  async function remove(filename: string): Promise<void> {
    setRemoving(filename)
    props.onError(null)
    try {
      await deleteProjector(props.model.id, filename)
      await props.onChanged()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setRemoving(null)
      setConfirming(null)
    }
  }

  async function chooseLocal(): Promise<void> {
    props.onError(null)
    try {
      const path = await window.brazier.selectFile('Choose an mmproj GGUF', [
        { name: 'GGUF projector', extensions: ['gguf'] }
      ])
      if (!path) return
      setAttaching(true)
      await attachProjector(props.model.id, path)
      await props.onChanged()
      setOpen(false)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setAttaching(false)
    }
  }

  const sharedNote =
    props.sharedWith > 1 ? ' Shared by every quant in this folder.' : ''

  return (
    <div className="library-projector">
      {projectors.length === 0 ? (
        <span className="library-projector-row">No vision projector</span>
      ) : (
        projectors.map((projector) => (
          <div className="library-projector-row" key={projector.name}>
            <span>
              {projector.name} · {formatBytes(projector.size_bytes)}
              {sharedNote}
            </span>
            {confirming === projector.name ? (
              <button
                type="button"
                className="chip-button danger"
                disabled={removing === projector.name}
                onClick={() => void remove(projector.name)}
              >
                {removing === projector.name ? (
                  <LoaderCircle className="spin" size={13} />
                ) : (
                  <Trash2 size={13} />
                )}
                Remove projector
              </button>
            ) : (
              <button
                type="button"
                className="chip-button subtle"
                title={`Remove this mmproj from disk.${sharedNote}`}
                onClick={() => setConfirming(projector.name)}
              >
                <Trash2 size={13} />
                Remove
              </button>
            )}
          </div>
        ))
      )}
      <div className="library-projector-actions">
        <button type="button" className="chip-button subtle" onClick={() => void loadOptions()}>
          {open ? 'Hide projectors' : 'Add projector'}
        </button>
        {props.allowLocalFile && (
          <button
            type="button"
            className="chip-button subtle"
            disabled={attaching}
            title="Copy an mmproj GGUF from this computer next to the model"
            onClick={() => void chooseLocal()}
          >
            {attaching ? <LoaderCircle className="spin" size={13} /> : <FolderOpen size={13} />}
            Choose file
          </button>
        )}
      </div>
      {queued && (
        <span className="library-projector-row">
          <LoaderCircle className="spin" size={13} /> Queued {basename(queued)}. Track it in the
          download tray.
        </span>
      )}
      {open && (
        <div className="library-projector-options">
          {loading && (
            <span className="library-projector-row">
              <LoaderCircle className="spin" size={13} /> Looking up projectors…
            </span>
          )}
          {!loading && !props.model.repo_id && (
            <span className="library-projector-row">
              Brazier cannot tell which Hugging Face repository this file came from.
              {props.allowLocalFile ? ' Choose the mmproj file on disk instead.' : ''}
            </span>
          )}
          {!loading && props.model.repo_id && options?.length === 0 && (
            <span className="library-projector-row">
              No other mmproj files were published
              {projectors.length > 0 ? ' besides the one already installed' : ''}.
            </span>
          )}
          {!loading &&
            options?.map((file) => (
              <div className="library-projector-row" key={file.path}>
                <span>
                  {basename(file.path)}
                  {file.size != null ? ` · ${formatBytes(file.size)}` : ''}
                </span>
                <button
                  type="button"
                  className="chip-button"
                  disabled={queued === file.path}
                  onClick={() => void download(file)}
                >
                  {queued === file.path ? (
                    <LoaderCircle className="spin" size={13} />
                  ) : (
                    <Download size={13} />
                  )}
                  Download
                </button>
              </div>
            ))}
        </div>
      )}
    </div>
  )
}
