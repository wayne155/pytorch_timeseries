import type { ViewOptions } from '../types'

interface ToolBarProps {
  viewOptions: ViewOptions
  allMetrics: string[]
  resultCount: number
  onViewChange: (v: ViewOptions) => void
  onExportCsv: () => void
  onRefresh?: () => void
}

export function ToolBar({
  viewOptions, allMetrics, resultCount, onViewChange, onExportCsv, onRefresh,
}: ToolBarProps) {
  const toggle = (patch: Partial<ViewOptions>) => onViewChange({ ...viewOptions, ...patch })

  const toggleMetric = (key: string) => {
    const cur = viewOptions.visibleMetrics
    toggle({
      visibleMetrics: cur.includes(key) ? cur.filter(k => k !== key) : [...cur, key],
    })
  }

  return (
    <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-200 bg-white flex-wrap">
      {/* Aggregate toggle */}
      <button
        className={`text-sm px-3 py-1 rounded border ${viewOptions.aggregate ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300 hover:border-gray-400'}`}
        onClick={() => toggle({ aggregate: !viewOptions.aggregate })}
      >
        {viewOptions.aggregate ? 'Aggregated' : 'Per seed'}
      </button>

      {/* Show std */}
      {viewOptions.aggregate && (
        <label className="flex items-center gap-1.5 text-sm text-gray-600 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={viewOptions.showStd}
            onChange={e => toggle({ showStd: e.target.checked })}
          />
          Show ±std
        </label>
      )}

      {/* Column visibility */}
      <div className="relative group">
        <button className="text-sm px-3 py-1 rounded border border-gray-300 hover:border-gray-400 bg-white text-gray-700">
          Columns ▾
        </button>
        <div className="absolute left-0 top-full mt-1 bg-white border border-gray-200 rounded shadow-lg p-2 z-10 hidden group-hover:block min-w-max">
          {allMetrics.map(k => (
            <label key={k} className="flex items-center gap-1.5 text-sm cursor-pointer hover:bg-gray-50 px-2 py-0.5 rounded">
              <input
                type="checkbox"
                checked={viewOptions.visibleMetrics.includes(k)}
                onChange={() => toggleMetric(k)}
              />
              {k}
            </label>
          ))}
        </div>
      </div>

      {/* Result count */}
      <span className="text-sm text-gray-500 ml-auto">{resultCount} results</span>

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
