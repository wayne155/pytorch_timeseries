// webapp/src/components/VariantBar.tsx
import type { TaskViewOptions } from '../types'

interface VariantBarProps {
  variants: string[]
  selectedVariant: string
  viewOptions: TaskViewOptions
  resultCount: number
  onVariantChange: (v: string) => void
  onOptionsChange: (o: TaskViewOptions) => void
  onExportCsv: () => void
  onRefresh?: () => void
}

export function VariantBar({
  variants, selectedVariant, viewOptions, resultCount,
  onVariantChange, onOptionsChange, onExportCsv, onRefresh,
}: VariantBarProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-white border-b border-gray-200 flex-wrap">
      {/* Variant selector */}
      {variants.length > 1 && (
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-gray-500">Variant:</span>
          <select
            className="text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-blue-400"
            value={selectedVariant}
            onChange={e => onVariantChange(e.target.value)}
          >
            {variants.map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>
      )}

      {/* ±std toggle */}
      <label className="flex items-center gap-1.5 text-sm text-gray-600 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={viewOptions.showStd}
          onChange={e => onOptionsChange({ ...viewOptions, showStd: e.target.checked })}
        />
        ±std
      </label>

      {/* Result count */}
      <span className="text-sm text-gray-400 ml-auto">{resultCount} models</span>

      {/* Export CSV */}
      <button
        className="text-sm px-3 py-1 rounded border border-gray-300 hover:border-gray-400 bg-white text-gray-700"
        onClick={onExportCsv}
      >
        Export CSV
      </button>

      {/* Refresh (live server only) */}
      {onRefresh && (
        <button
          className="text-sm px-3 py-1 rounded border border-gray-300 hover:border-gray-400 bg-white text-gray-700"
          onClick={onRefresh}
        >
          ↺ Refresh
        </button>
      )}
    </div>
  )
}
