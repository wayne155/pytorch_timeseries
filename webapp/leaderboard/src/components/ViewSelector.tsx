// webapp/src/components/ViewSelector.tsx
import type { ViewData } from '../types'

interface ViewSelectorProps {
  views: ViewData[]
  selectedId: string
  onSelect: (id: string) => void
}

export function ViewSelector({ views, selectedId, onSelect }: ViewSelectorProps) {
  return (
    <div className="flex gap-0 border-b border-line bg-surface-panel px-4 overflow-x-auto">
      {views.map((v, i) => {
        const active = v.id === selectedId
        return (
          <button
            key={v.id}
            onClick={() => onSelect(v.id)}
            className={`group px-4 py-2.5 text-[13px] whitespace-nowrap border-b-[3px] -mb-px transition-colors tracking-wide ${
              active
                ? 'border-phosphor bg-phosphor-dim text-phosphor-bright font-semibold'
                : 'border-transparent text-ink-dim hover:text-ink hover:border-line-strong'
            }`}
          >
            <span className={`mr-1.5 text-[11px] ${active ? 'text-phosphor' : 'text-ink-faint group-hover:text-ink-dim'}`}>
              {String(i + 1).padStart(2, '0')}
            </span>
            {v.display_name}
          </button>
        )
      })}
    </div>
  )
}
