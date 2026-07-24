import { Check, Globe, Wrench } from 'lucide-react'
import type { BundledTool } from '../api'

type ToolsMenuProps = {
  tools: BundledTool[]
  enabled: string[]
  disabled: boolean
  onToggle: (name: string, on: boolean) => void
  onSetAll: (names: string[], on: boolean) => void
  onClose: () => void
}

type ToolGroup = {
  key: string
  label: string
  tools: BundledTool[]
}

/** Group bundled tools first, then one group per MCP server. */
function groupTools(tools: BundledTool[]): ToolGroup[] {
  const bundled: BundledTool[] = []
  const byServer = new Map<string, BundledTool[]>()
  for (const tool of tools) {
    if (tool.source === 'mcp' || tool.name.startsWith('mcp/')) {
      const server = tool.name.split('/')[1] ?? 'mcp'
      const list = byServer.get(server) ?? []
      list.push(tool)
      byServer.set(server, list)
    } else {
      bundled.push(tool)
    }
  }
  const groups: ToolGroup[] = []
  if (bundled.length > 0) groups.push({ key: 'bundled', label: 'Bundled', tools: bundled })
  for (const [server, list] of byServer) {
    groups.push({ key: `mcp:${server}`, label: `MCP · ${server}`, tools: list })
  }
  return groups
}

/**
 * Per-tool selection popover anchored above the composer's tool button. Lets
 * the user offer a subset of bundled and MCP tools to the model instead of the
 * previous all-or-nothing toggle.
 */
export function ToolsMenu({
  tools,
  enabled,
  disabled,
  onToggle,
  onSetAll,
  onClose
}: ToolsMenuProps): React.JSX.Element {
  const groups = groupTools(tools)
  const allNames = tools.map((tool) => tool.name)
  const allOn = allNames.length > 0 && allNames.every((name) => enabled.includes(name))

  return (
    <>
      <div className="tool-menu-backdrop" onMouseDown={onClose} />
      <div className="tool-menu-popover" onMouseDown={(event) => event.stopPropagation()}>
        <div className="tool-menu-head">
          <span className="popover-title">Tools</span>
          <div className="tool-menu-bulk">
            <button
              type="button"
              disabled={disabled || allOn}
              onClick={() => onSetAll(allNames, true)}
            >
              All
            </button>
            <button
              type="button"
              disabled={disabled || enabled.length === 0}
              onClick={() => onSetAll(allNames, false)}
            >
              None
            </button>
          </div>
        </div>
        {disabled && (
          <p className="tool-menu-note">This model does not advertise tool support.</p>
        )}
        {tools.length === 0 && (
          <p className="tool-menu-note">No tools available. Add MCP servers in Manage.</p>
        )}
        {groups.map((group) => (
          <div className="tool-menu-group" key={group.key}>
            <div className="section-label">{group.label}</div>
            {group.tools.map((tool) => {
              const on = enabled.includes(tool.name)
              return (
                <button
                  key={tool.name}
                  type="button"
                  className={on ? 'tool-menu-item on' : 'tool-menu-item'}
                  disabled={disabled}
                  onClick={() => onToggle(tool.name, !on)}
                >
                  <span className={on ? 'tool-menu-check on' : 'tool-menu-check'}>
                    {on && <Check size={12} />}
                  </span>
                  <span className="tool-menu-item-body">
                    <strong>
                      {tool.title || tool.name}
                      {tool.network && (
                        <span className="tool-menu-net" title="Uses the network">
                          <Globe size={11} />
                        </span>
                      )}
                    </strong>
                    {tool.description && <span>{tool.description}</span>}
                  </span>
                </button>
              )
            })}
          </div>
        ))}
        <button className="popover-footer-action" onClick={onClose}>
          <Wrench size={14} />
          Done
        </button>
      </div>
    </>
  )
}
