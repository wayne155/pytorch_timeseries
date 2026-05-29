import { useState, useMemo, useCallback } from 'react'
import { useLeaderboard } from './hooks/useLeaderboard'
import { useFiltered } from './hooks/useFiltered'
import { FilterSidebar } from './components/FilterSidebar'
import { ToolBar } from './components/ToolBar'
import { LeaderboardTable } from './components/LeaderboardTable'
import type { Filters, ViewOptions, DisplayRow } from './types'

function rowsToCsv(rows: DisplayRow[], visibleMetrics: string[]): string {
  const metricHeaders = visibleMetrics.flatMap(k => [`${k}_mean`, `${k}_std`])
  const header = ['model', 'task', 'dataset', 'hparams', 'seeds', 'source', ...metricHeaders]
  const lines = rows.map(r => {
    const metricCells = visibleMetrics.flatMap(k => {
      const m = r.metrics[k]
      if (m === undefined) return ['', '']
      if (typeof m === 'number') return [String(m), '']
      return [String(m.mean), String(m.std)]
    })
    return [
      r.model, r.task, r.dataset,
      JSON.stringify(r.hparams), String(r.num_seeds), r.source_type,
      ...metricCells,
    ].map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')
  })
  return [header.join(','), ...lines].join('\n')
}

function downloadCsv(content: string, filename: string) {
  const blob = new Blob([content], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

const isLiveServer = !import.meta.env.PROD || window.location.port !== ''

export default function App() {
  const { data, error, loading, refresh } = useLeaderboard()

  const [filters, setFilters] = useState<Filters>({
    task: null, datasets: [], models: [], hparams: {},
  })

  const allMetrics = useMemo(() => {
    if (!data) return []
    const keys = new Set<string>()
    data.entries.forEach(e => Object.keys(e.metrics).forEach(k => keys.add(k)))
    return Array.from(keys).sort()
  }, [data])

  const [viewOptions, setViewOptions] = useState<ViewOptions>({
    aggregate: true, showStd: false,
    visibleMetrics: [],
    sortColumn: null, sortDirection: 'asc',
  })

  const effectiveOptions = useMemo(() => ({
    ...viewOptions,
    visibleMetrics: viewOptions.visibleMetrics.length > 0
      ? viewOptions.visibleMetrics
      : allMetrics,
  }), [viewOptions, allMetrics])

  const rows = useFiltered(data?.entries ?? [], filters, effectiveOptions)

  const handleHparamClick = useCallback((key: string, value: string) => {
    setFilters(f => ({ ...f, hparams: { ...f.hparams, [key]: value } }))
  }, [])

  const handleExportCsv = useCallback(() => {
    downloadCsv(rowsToCsv(rows, effectiveOptions.visibleMetrics), 'leaderboard.csv')
  }, [rows, effectiveOptions.visibleMetrics])

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center text-gray-500">
      Loading leaderboard data…
    </div>
  )

  if (error) return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-red-600 text-center">
        <p className="font-semibold">Failed to load leaderboard_data.json</p>
        <p className="text-sm mt-1">{error}</p>
        <p className="text-sm mt-2 text-gray-500">
          Run: <code className="bg-gray-100 px-1 rounded">python scripts/build_leaderboard.py</code>
        </p>
      </div>
    </div>
  )

  if (!data) return null

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-3">
        <h1 className="text-lg font-bold text-gray-900">torch-timeseries Leaderboard</h1>
        <span className="text-xs text-gray-400">
          Generated {new Date(data.generated_at).toLocaleString()}
        </span>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <FilterSidebar schema={data.schema} filters={filters} onChange={setFilters} />

        <main className="flex-1 flex flex-col overflow-hidden">
          <ToolBar
            viewOptions={effectiveOptions}
            allMetrics={allMetrics}
            resultCount={rows.length}
            onViewChange={setViewOptions}
            onExportCsv={handleExportCsv}
            onRefresh={isLiveServer ? refresh : undefined}
          />
          <div className="flex-1 overflow-auto">
            <LeaderboardTable
              rows={rows}
              viewOptions={effectiveOptions}
              onSortChange={(col, dir) => setViewOptions(v => ({ ...v, sortColumn: col, sortDirection: dir }))}
              onHparamClick={handleHparamClick}
            />
          </div>
        </main>
      </div>
    </div>
  )
}
