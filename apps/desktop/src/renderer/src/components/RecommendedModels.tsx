/**
 * One recommended model per thing you might want to do, sized for this machine.
 *
 * The first thing a new installation asks of someone is the hardest question in
 * the whole application: which of several thousand models, at which of a dozen
 * quantisations. This answers it — one card per category, with the quant already
 * chosen against how much memory the machine actually has, and a button that
 * downloads exactly that.
 *
 * The same cards appear in the welcome flow and in Manage, because "set this up
 * for me" and "show me that again later" are the same request.
 */

import {
  AlertTriangle,
  Check,
  Download,
  ExternalLink,
  Hammer,
  LoaderCircle
} from 'lucide-react'
import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react'

import {
  addHfAccessRequest,
  downloadModel,
  downloadPersonaplexModel,
  buildRuntime,
  activateRuntime,
  formatBytes,
  installSdcppBundle,
  listSdcppBundles,
  listRuntimes,
  huggingFaceTokenStatus,
  recordRecommendationInstall,
  setHuggingFaceToken,
  listRecommendationSetups,
  startRecommendationSetup,
  resolveBundleVariants,
  type BundleRecommendation,
  type ProgressEvent,
  type RecommendationCategory,
  type RecommendationSetup,
  type Recommendations,
  type RepoRecommendation,
  type VoiceRecommendationModel
} from '../api'

export const CATEGORY_LABELS: Record<RecommendationCategory, string> = {
  text: 'Chat',
  agent: 'Agent',
  image: 'Image generation',
  video: 'Video generation',
  voice: 'Voice',
  computer_use: 'Computer use'
}

const CATEGORY_BLURBS: Record<RecommendationCategory, string> = {
  text: 'Conversation, writing, and questions.',
  agent: 'Edits files and runs commands in a folder you choose.',
  image: 'Makes images from a prompt.',
  video: 'Makes short clips from a prompt.',
  voice: 'Speak and be answered out loud.',
  computer_use: 'Operates a browser or desktop from screenshots.'
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function formatContextTokens(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`
  return `${Math.round(tokens / 1000)}k`
}

function progressText(event: ProgressEvent | null): string {
  if (!event) return 'Starting…'
  if (event.phase === 'download' && event.total) {
    const percent = Math.round(
      event.percent ?? ((event.bytes ?? 0) / event.total) * 100
    )
    return `Downloading ${percent}%`
  }
  return event.message ?? 'Working…'
}

type InstallState = { busy: boolean; done: boolean; progress: ProgressEvent | null }

const IDLE: InstallState = { busy: false, done: false, progress: null }

/** Shell shared by every card, so they read as one set rather than four designs. */
function Card(props: {
  category: RecommendationCategory
  categoryLabel?: string
  blurb?: string
  title: string
  summary?: string | null
  meta?: string | null
  notes?: Array<{ tone: 'warn' | 'plain'; text: string }>
  action: React.ReactNode
}): React.JSX.Element {
  return (
    <article className="recommendation-card">
      <div className="recommendation-card-body">
        <div className="recommendation-card-head">
          <span className="recommendation-category">
            {props.categoryLabel ?? CATEGORY_LABELS[props.category]}
          </span>
          <strong>{props.title}</strong>
        </div>
        {props.summary ? (
          <p className="recommendation-summary">{props.summary}</p>
        ) : (
          <p className="recommendation-blurb">
            {props.blurb ?? CATEGORY_BLURBS[props.category]}
          </p>
        )}
        {props.meta ? <p className="recommendation-meta">{props.meta}</p> : null}
        {(props.notes ?? []).map((note, index) => (
          <p
            key={`${note.tone}-${index}`}
            className={note.tone === 'warn' ? 'recommendation-note warn' : 'recommendation-note'}
          >
            {note.tone === 'warn' ? <AlertTriangle size={12} /> : null}
            {note.text}
          </p>
        ))}
      </div>
      <div className="recommendation-card-action">{props.action}</div>
    </article>
  )
}

/**
 * Opens a gated model's Hugging Face page to request access, and records the
 * repository so the settings queue can watch for the grant.
 */
function RequestAccessLink({ repoId }: { repoId: string }): React.JSX.Element {
  return (
    <a
      className="chip-button"
      href={`https://huggingface.co/${repoId}`}
      target="_blank"
      rel="noreferrer"
      title={`Request access to ${repoId}`}
      onClick={() => {
        void addHfAccessRequest(repoId).catch(() => {})
      }}
    >
      <ExternalLink size={13} /> Request access
    </a>
  )
}

function InstallButton(props: {
  state: InstallState
  label: string
  disabled?: boolean
  onClick: () => void
}): React.JSX.Element {
  if (props.state.done) {
    return (
      <button type="button" className="chip-button selected" disabled>
        <Check size={13} /> Installed
      </button>
    )
  }
  if (props.state.busy) {
    return (
      <button type="button" className="chip-button" disabled>
        <LoaderCircle className="spin" size={13} />
        {progressText(props.state.progress)}
      </button>
    )
  }
  return (
    <button
      type="button"
      className="chip-button"
      disabled={props.disabled}
      onClick={props.onClick}
    >
      <Download size={13} /> {props.label}
    </button>
  )
}

type Props = {
  recommendations: Recommendations
  /** Categories to show. Omitted shows every category with a recommendation. */
  categories?: RecommendationCategory[]
  /** Called after anything installs, so the caller can refresh its model list. */
  onInstalled?: () => void
  onError?: (message: string | null) => void
  /** Offer to build the PersonaPlex runtime, which voice also needs. */
  onOpenRuntimes?: () => void
}

export function RecommendedModels(props: Props): React.JSX.Element {
  const { recommendations, onInstalled, onError } = props
  const [states, setStates] = useState<Record<string, InstallState>>({})
  const [includeCompanions, setIncludeCompanions] = useState<Record<string, boolean>>({})
  const [includeDrafts, setIncludeDrafts] = useState<Record<string, boolean>>({})
  const [setups, setSetups] = useState<RecommendationSetup[]>([])
  const [hfTokenSource, setHfTokenSource] = useState('none')
  const [hfTokenDraft, setHfTokenDraft] = useState('')
  const [savingHfToken, setSavingHfToken] = useState(false)
  const announcedSetups = useRef(new Set<string>())

  const categories = props.categories ?? (Object.keys(recommendations.categories) as RecommendationCategory[])
  const hasGatedCategory = categories.some((category) => {
    const entry = recommendations.categories[category as keyof typeof recommendations.categories]
    return entry?.gated === true
  })
  const hasGatedRecommendation =
    hasGatedCategory ||
    (categories.includes('agent') &&
      recommendations.agent_options?.some((entry) => entry.gated === true) === true) ||
    (categories.includes('voice') &&
      recommendations.voice?.models.some((entry) => entry.gated === true) === true)

  useEffect(() => {
    const refresh = (): void => {
      void listRecommendationSetups().then(setSetups).catch(() => {})
    }
    refresh()
    const timer = window.setInterval(refresh, 1500)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!hasGatedRecommendation) return
    void huggingFaceTokenStatus()
      .then((status) => setHfTokenSource(status.source))
      .catch(() => setHfTokenSource('none'))
  }, [hasGatedRecommendation])

  useEffect(() => {
    for (const setup of setups) {
      if (setup.status === 'completed' && !announcedSetups.current.has(setup.id)) {
        announcedSetups.current.add(setup.id)
        onInstalled?.()
      }
    }
  }, [setups, onInstalled])

  const setState = useCallback((key: string, patch: Partial<InstallState>) => {
    setStates((current) => ({ ...current, [key]: { ...(current[key] ?? IDLE), ...patch } }))
  }, [])

  async function saveHubToken(event: FormEvent): Promise<void> {
    event.preventDefault()
    setSavingHfToken(true)
    onError?.(null)
    try {
      await setHuggingFaceToken(hfTokenDraft.trim())
      setHfTokenDraft('')
      const status = await huggingFaceTokenStatus()
      setHfTokenSource(status.source)
    } catch (cause) {
      onError?.(errorText(cause))
    } finally {
      setSavingHfToken(false)
    }
  }

  // A category already set up through this flow opens as installed rather than
  // inviting the same download a second time.
  useEffect(() => {
    const installed = recommendations.state.installed
    setStates((current) => {
      const next = { ...current }
      for (const [category, record] of Object.entries(installed)) {
        if (record.recommendation_id) {
          next[category] = { ...(next[category] ?? IDLE), done: true }
        }
      }
      return next
    })
  }, [recommendations.state.installed])

  async function run(
    key: string,
    categories: RecommendationCategory[],
    recommendationId: string,
    work: (onProgress: (event: ProgressEvent) => void) => Promise<string | undefined>
  ): Promise<void> {
    setState(key, { busy: true, progress: null })
    onError?.(null)
    try {
      const modelId = await work((event) => setState(key, { progress: event }))
      await Promise.all(
        categories.map((category) =>
          recordRecommendationInstall(category, recommendationId, modelId)
        )
      )
      setState(key, { busy: false, done: true })
      onInstalled?.()
    } catch (cause) {
      setState(key, { busy: false })
      onError?.(errorText(cause))
    }
  }

  function repoCard(
    categories: Array<'text' | 'agent' | 'computer_use'>,
    entry: RepoRecommendation
  ): React.JSX.Element {
    const combined = categories.length === 2
    const category = categories[0]
    const key = combined ? 'text-agent' : category
    const recordedForAll = categories.every(
      (entryCategory) =>
        recommendations.state.installed[entryCategory]?.recommendation_id === entry.id
    )
    const setup = setups.find(
      (candidate) =>
        candidate.recommendation_id === entry.id &&
        categories.every((category) => candidate.categories.includes(category)) &&
        ['pending', 'running', 'paused'].includes(candidate.status)
    )
    const state = recordedForAll
      ? { ...(states[key] ?? IDLE), done: true }
      : setup
        ? {
            ...(states[key] ?? IDLE),
            busy: true,
            progress: setup.status === 'paused'
              ? { phase: 'paused', message: 'Paused' }
              : { phase: 'setup', message: 'Installing…' }
          }
        : states[key] ?? IDLE
    const files = entry.files ?? []
    const notes: Array<{ tone: 'warn' | 'plain'; text: string }> = []
    if (entry.substituted) notes.push({ tone: 'plain', text: entry.substituted })
    if (entry.unresolved) notes.push({ tone: 'warn', text: entry.unresolved })
    if (entry.gated) {
      notes.push({
        tone: 'warn',
        text: 'Gated on Hugging Face. Accept its terms there, then add a token below.'
      })
    }
    if (entry.tight) {
      notes.push({
        tone: 'warn',
        text: 'A tight fit for this machine’s memory. Expect it to be slow, with little room for long conversations.'
      })
    }
    const companions = entry.companion_files ?? []
    const computerUseModel = categories.includes('computer_use')
    const includeCompanion = computerUseModel || (includeCompanions[key] ?? true)
    const drafts = entry.draft_files ?? []
    const includeDraft = includeDrafts[key] ?? true
    if (computerUseModel && entry.context_tokens) {
      notes.push({
        tone: 'plain',
        text: `Uses ${formatContextTokens(entry.context_tokens)} of context by default so long screenshot sessions fit.`
      })
    }
    if (entry.runtime_build) {
      notes.push({
        tone: 'plain',
        text: `Builds ${entry.runtime_build.label} automatically.`
      })
    }
    if (entry.unresolved_companions) {
      notes.push({ tone: 'warn', text: entry.unresolved_companions })
    }
    if (entry.unresolved_drafts) {
      notes.push({ tone: 'warn', text: entry.unresolved_drafts })
    }

    const meta = entry.bytes
      ? `${entry.quant ? `${entry.quant} · ` : ''}${formatBytes(entry.bytes)} · ${entry.repo_id}`
      : entry.repo_id

    return (
      <Card
        key={key}
        category={category}
        categoryLabel={combined ? 'Chat / Agent' : undefined}
        blurb={
          combined
            ? 'Conversation, writing, questions, editing files, and running commands.'
            : undefined
        }
        title={entry.label}
        summary={entry.summary}
        meta={meta}
        notes={notes}
        action={
          <div className="recommendation-actions">
            {companions.length > 0 && !computerUseModel ? (
              <label className="recommendation-companion-choice">
                <input
                  type="checkbox"
                  checked={includeCompanion}
                  disabled={state.busy || state.done}
                  onChange={(event) =>
                    setIncludeCompanions((current) => ({ ...current, [key]: event.target.checked }))
                  }
                />
                Image input
              </label>
            ) : null}
            {drafts.length > 0 ? (
              <label
                className="recommendation-companion-choice"
                title="A small draft model that speeds up generation"
              >
                <input
                  type="checkbox"
                  checked={includeDraft}
                  disabled={state.busy || state.done}
                  onChange={(event) =>
                    setIncludeDrafts((current) => ({ ...current, [key]: event.target.checked }))
                  }
                />
                Faster decoding
              </label>
            ) : null}
            <InstallButton
              state={state}
              label="Install"
              disabled={Boolean(entry.unresolved) || files.length === 0 || (entry.gated && hfTokenSource === 'none')}
              onClick={() =>
                void (async () => {
                  setState(key, { busy: true, progress: null })
                  onError?.(null)
                  try {
                    await startRecommendationSetup({
                      recommendation_id: entry.id,
                      categories,
                      required_bytes: entry.bytes ?? 0,
                      build: entry.runtime_build
                        ? { ...entry.runtime_build, jobs: 0 }
                        : undefined,
                      works: [
                        ...files,
                        ...(includeCompanion ? companions : []),
                        ...(includeDraft ? drafts : [])
                      ].map((filename) => ({
                        kind: 'gguf' as const,
                        repo_id: entry.repo_id,
                        filename,
                        revision: 'main',
                        engine: 'llama.cpp' as const
                      }))
                    })
                    // Activity owns the rest of the work; the card is no
                    // longer allowed to create a second local sequence.
                    setState(key, { busy: false })
                  } catch (cause) {
                    setState(key, { busy: false })
                    onError?.(errorText(cause))
                  }
                })()
              }
            />
            {entry.gated ? <RequestAccessLink repoId={entry.repo_id} /> : null}
          </div>
        }
      />
    )
  }

  function bundleCard(
    category: 'image' | 'video',
    entry: BundleRecommendation
  ): React.JSX.Element {
    const parts =
      entry.parts && entry.parts.length > 0
        ? entry.parts
        : entry.bundle_id
          ? [{ bundle_id: entry.bundle_id, role: 'model', label: entry.label }]
          : []
    const split = (entry.parts?.length ?? 0) > 1
    const notes: Array<{ tone: 'warn' | 'plain'; text: string }> = []
    if (entry.unresolved) notes.push({ tone: 'warn', text: entry.unresolved })
    if (entry.gated) {
      notes.push({
        tone: 'warn',
        text: 'Includes gated Hugging Face files. Accept their terms there, then add a token below.'
      })
    }
    if (split) {
      notes.push({
        tone: 'plain',
        text: 'Two models: one makes clips from text, the other animates a picture you supply. Install either or both.'
      })
    }

    return (
      <Card
        key={category}
        category={category}
        title={entry.label}
        summary={entry.summary}
        meta={entry.variant ? `${entry.variant} · ${parts.map((p) => p.bundle_id).join(', ')}` : null}
        notes={notes}
        action={
          <div className="recommendation-actions">
            {parts.map((part) => {
              const key = `${category}:${part.bundle_id}`
              const state = states[key] ?? IDLE
              return (
                <InstallButton
                  key={key}
                  state={state}
                  label={split ? `Install ${part.role}` : 'Install'}
                  disabled={Boolean(entry.unresolved) || (entry.gated && hfTokenSource === 'none')}
                  onClick={() =>
                    void run(key, [category], entry.id, async (onProgress) => {
                      const bundle = (await listSdcppBundles()).find(
                        (candidate) => candidate.id === part.bundle_id
                      )
                      if (!bundle) {
                        throw new Error(`Recommended bundle ${part.bundle_id} is unavailable.`)
                      }
                      const wantedVariant = part.variant ?? entry.variant
                      const choices = wantedVariant
                        ? Object.fromEntries(
                            bundle.components.flatMap((component, index) =>
                              component.variants?.some((variant) => variant.label === wantedVariant)
                                ? [[index, wantedVariant]]
                                : []
                            )
                          )
                        : {}
                      const result = await installSdcppBundle(
                        { bundle: resolveBundleVariants(bundle, choices) },
                        onProgress
                      )
                      return result.model_id
                    })
                  }
                />
              )
            })}
            {entry.gated
              ? (entry.gated_repos ?? []).map((repoId) => (
                  <RequestAccessLink key={repoId} repoId={repoId} />
                ))
              : null}
          </div>
        }
      />
    )
  }

  function voiceCard(
    summary: string | null | undefined,
    models: VoiceRecommendationModel[]
  ): React.JSX.Element {
    return (
      <Card
        key="voice"
        category="voice"
        title="PersonaPlex and a recogniser"
        summary={summary}
        notes={[
          {
            tone: 'plain',
            text: 'Also needs the PersonaPlex runtime, which is built from source.'
          }
        ]}
        action={
          <div className="recommendation-actions">
            {models.map((model) => {
              const key = `voice:${model.id}`
              const state = states[key] ?? IDLE
              return (
                <InstallButton
                  key={key}
                  state={state}
                  label={model.label}
                  onClick={() =>
                    void run(key, ['voice'], model.id, async (onProgress) => {
                      if (model.kind === 'personaplex') {
                        const result = await downloadPersonaplexModel(model.repo_id, onProgress)
                        return result.model_id
                      }
                      const result = await downloadModel(
                        model.repo_id,
                        model.filename ?? '',
                        onProgress,
                        'main',
                        'whisper.cpp'
                      )
                      return result.model_id
                    })
                  }
                />
              )
            })}
            {props.onOpenRuntimes ? (
              <button type="button" className="chip-button subtle" onClick={props.onOpenRuntimes}>
                <Hammer size={13} /> Build runtime
              </button>
            ) : null}
            {models
              .filter((model) => model.gated)
              .map((model) => (
                <RequestAccessLink key={model.id} repoId={model.repo_id} />
              ))}
          </div>
        }
      />
    )
  }

  const wanted = props.categories ?? (['text', 'agent', 'image', 'video', 'voice', 'computer_use'] as const)
  const cards: React.JSX.Element[] = []
  const wantsText = wanted.includes('text')
  const wantsAgent = wanted.includes('agent')
  const text = recommendations.categories.text
  const agent = recommendations.categories.agent
  const agentOptions = recommendations.agent_options?.length
    ? recommendations.agent_options
    : agent
      ? [agent]
      : []
  const sharedTextAgent =
    wantsText &&
    wantsAgent &&
    text &&
    agentOptions[0] &&
    text.id === agentOptions[0].id &&
    text.repo_id === agentOptions[0].repo_id
  if (sharedTextAgent) cards.push(repoCard(['text', 'agent'], text))
  for (const category of wanted) {
    if (category === 'voice') {
      const voice = recommendations.voice
      if (voice) cards.push(voiceCard(voice.summary, voice.models))
      continue
    }
    if (category === 'text' || category === 'agent') {
      if (sharedTextAgent) continue
      if (category === 'agent') {
        for (const entry of agentOptions) cards.push(repoCard(['agent'], entry))
        continue
      }
      const entry = recommendations.categories[category]
      if (entry) cards.push(repoCard([category], entry))
      continue
    }
    if (category === 'computer_use') {
      const entry = recommendations.categories.computer_use
      if (entry) cards.push(repoCard(['computer_use'], entry))
      continue
    }
    const entry = recommendations.categories[category]
    if (entry) cards.push(bundleCard(category, entry))
  }

  if (recommendations.reason) {
    return (
      <div className="manage-placeholder compact">
        <AlertTriangle size={16} />
        {recommendations.reason}
      </div>
    )
  }

  return (
    <div className="recommendation-list">
      {recommendations.memory_bytes ? (
        <p className="recommendation-tier">
          For {formatBytes(recommendations.memory_bytes)} of{' '}
          {recommendations.memory_source === 'vram' ? 'video memory' : 'memory'}
        </p>
      ) : null}
      {hasGatedRecommendation ? (
        <form className="build-form" onSubmit={(event) => void saveHubToken(event)}>
          <label>
            <span className="label-with-link">
              Hugging Face token
              <a
                className="inline-link"
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noreferrer"
              >
                Get a token
              </a>
            </span>
            <input
              type="password"
              autoComplete="off"
              value={hfTokenDraft}
              onChange={(event) => setHfTokenDraft(event.target.value)}
              placeholder={
                hfTokenSource === 'environment'
                  ? 'Using HF_TOKEN from your environment'
                  : hfTokenSource === 'stored'
                    ? 'Saved. Paste a new one to replace it'
                    : 'hf_…'
              }
            />
          </label>
          <p className="model-help">
            Accept the model’s terms on Hugging Face first. Kept on this device.
          </p>
          <div className="build-form-actions">
            <button
              className="secondary-action"
              type="submit"
              disabled={savingHfToken || !hfTokenDraft.trim()}
            >
              {savingHfToken ? <LoaderCircle className="spin" size={14} /> : 'Save'}
            </button>
          </div>
        </form>
      ) : null}
      {cards.length === 0 ? (
        <div className="manage-placeholder compact">
          No recommendations for what you picked.
        </div>
      ) : (
        cards
      )}
    </div>
  )
}
