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
    <div className="flex items-center gap-4 px-5 py-2 border-b border-line bg-surface-panel flex-wrap text-[13px]">
      {/* Variant selector — segmented control */}
      {variants.length > 1 && (
        <div className="flex items-center gap-2">
          <span className="text-[11px] uppercase tracking-[0.2em] text-ink-faint">variant</span>
          <div className="flex rounded border border-line-strong overflow-hidden">
            {variants.map(v => (
              <button
                key={v}
                onClick={() => onVariantChange(v)}
                className={`px-3 py-1 transition-colors ${
                  v === selectedVariant
                    ? 'bg-ink text-[#FFFFFF] font-semibold'
                    : 'text-ink-dim hover:text-ink hover:bg-surface-raised'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ±std toggle */}
      <label className="flex items-center gap-2 text-ink cursor-pointer select-none transition-colors">
        <input
          type="checkbox"
          checked={viewOptions.showStd}
          onChange={e => onOptionsChange({ ...viewOptions, showStd: e.target.checked })}
          className="instrument-check"
        />
        ±std
      </label>

      {/* Result count */}
      <span className="ml-auto text-ink-faint tabular-nums">
        {resultCount} {resultCount === 1 ? 'model' : 'models'}
      </span>

      {/* Export CSV */}
      <button
        className="px-3.5 py-1.5 rounded border-2 border-ink/50 bg-white text-ink font-medium shadow-sm hover:border-phosphor hover:text-phosphor-bright hover:bg-phosphor-dim transition-colors"
        onClick={onExportCsv}
      >
        ↓ Export CSV
      </button>

      {/* Refresh (live server only) */}
      {onRefresh && (
        <button
          className="px-3.5 py-1.5 rounded border-2 border-ink/50 bg-white text-ink font-medium shadow-sm hover:border-phosphor hover:text-phosphor-bright hover:bg-phosphor-dim transition-colors"
          onClick={onRefresh}
        >
          ↺ Refresh
        </button>
      )}
    </div>
  )
}
