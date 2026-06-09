// webapp/src/components/ViewSelector.tsx
import type { ViewData } from '../types'

interface ViewSelectorProps {
  views: ViewData[]
  selectedId: string
  onSelect: (id: string) => void
}

export function ViewSelector({ views, selectedId, onSelect }: ViewSelectorProps) {
  return (
    <div className="flex gap-0 border-b border-gray-200 bg-white px-4 overflow-x-auto">
      {views.map(v => (
        <button
          key={v.id}
          onClick={() => onSelect(v.id)}
          className={`px-4 py-2.5 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
            v.id === selectedId
              ? 'border-blue-600 text-blue-700'
              : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
          }`}
        >
          {v.display_name}
        </button>
      ))}
    </div>
  )
}
