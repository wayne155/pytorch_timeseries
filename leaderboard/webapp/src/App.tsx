import { useState, useEffect } from 'react'
import { useLeaderboard } from './hooks/useLeaderboard'
import { useTaskView } from './hooks/useTaskView'
import type { ColumnDef } from './hooks/useTaskView'
import { ViewSelector } from './components/ViewSelector'
import { VariantBar } from './components/VariantBar'
import { DatasetSelector } from './components/DatasetSelector'
import { TaskTable } from './components/TaskTable'
import type { TaskTableRow, TaskViewOptions } from './types'

function rowsToCsv(rows: TaskTableRow[], columnDefs: ColumnDef[]): string {
  const headers = ['Model', ...columnDefs.map(c => c.label)]
  const lines = rows.map(row => {
    const cells = columnDefs.map(col => {
      const subcol = row.columns[col.id]
      if (!subcol) return ''
      // Use first available metric's mean
      const firstKey = Object.keys(subcol)[0]
      return firstKey ? String(subcol[firstKey].mean) : ''
    })
    return [row.model, ...cells].map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')
  })
  return [headers.join(','), ...lines].join('\n')
}

function downloadCsv(content: string, filename: string) {
  const blob = new Blob([content], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

const isLiveServer = !import.meta.env.PROD || window.location.port !== ''

/** Decorative time-series trace: solid history + dashed forecast. */
function SignalTrace() {
  return (
    <svg
      className="pointer-events-none absolute inset-x-0 bottom-0 h-16 w-full opacity-60"
      viewBox="0 0 900 64"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <polyline
        className="trace-history"
        points="0,44 40,40 70,46 110,30 150,38 190,22 230,34 270,18 310,30 350,12 390,26 430,20 470,32 510,16 550,28 580,24 600,30"
      />
      <polyline
        className="trace-forecast"
        points="600,30 640,20 680,28 720,14 760,24 800,10 840,20 880,8 900,14"
      />
      <circle cx="600" cy="30" r="2.5" fill="#c2780c" />
    </svg>
  )
}

export default function App() {
  const { data, error, loading, refresh } = useLeaderboard()

  const [selectedViewId, setSelectedViewId] = useState<string | null>(null)
  const [selectedVariant, setSelectedVariant] = useState<string>('')
  const [selectedDataset, setSelectedDataset] = useState<string>('All')
  const [viewOptions, setViewOptions] = useState<TaskViewOptions>({
    showStd: false,
    sortColumn: null,
    sortMetric: null,
    sortDirection: 'asc',
  })

  // Derive selectedView
  const selectedView = data?.views.find(v => v.id === selectedViewId) ?? data?.views[0] ?? null

  // Reset dataset and variant when view changes via data load (not explicit tab click)
  useEffect(() => {
    setSelectedDataset('All')
    setSelectedVariant('')
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedView?.id])

  // useTaskView must be called unconditionally (hooks rules)
  const { rows, columnDefs } = useTaskView(
    selectedView,
    selectedVariant || (selectedView?.variants[0] ?? ''),
    selectedDataset,
    viewOptions,
  )

  function handleSelectView(id: string) {
    setSelectedViewId(id)
    setSelectedDataset('All')
    setSelectedVariant('')
  }

  function handleVariantChange(v: string) {
    setSelectedVariant(v)
  }

  function handleSortChange(column: string, metric: string, dir: 'asc' | 'desc') {
    setViewOptions(prev => ({ ...prev, sortColumn: column, sortMetric: metric, sortDirection: dir }))
  }

  function handleExportCsv() {
    if (!selectedView) return
    const variant = selectedVariant || selectedView.variants[0]
    const filename = `${selectedView.id}_${variant}_${selectedDataset}.csv`
    downloadCsv(rowsToCsv(rows, columnDefs), filename)
  }

  const generatedAt = data?.generated_at
    ? new Date(data.generated_at).toISOString().replace('T', ' ').slice(0, 16) + ' UTC'
    : null

  return (
    <div className="min-h-screen flex flex-col font-mono">
      {/* Masthead */}
      <header className="relative overflow-hidden border-b border-line bg-surface-panel">
        <SignalTrace />
        <div className="relative flex items-end justify-between px-6 pt-5 pb-4">
          <div className="flex items-end gap-3.5">
            <img src="/favicon.svg" alt="torch-timeseries logo" className="h-12 w-12" />
            <div>
              <div className="text-[11px] uppercase tracking-[0.3em] text-phosphor mb-1">
                torch-timeseries
              </div>
              <h1 className="font-display italic text-4xl leading-none text-ink">
                Leaderboard
              </h1>
            </div>
          </div>
          <div className="hidden sm:flex flex-col items-end gap-1 pb-1 text-xs text-ink-dim">
            <span className="flex items-center gap-1.5">
              <span className="live-dot inline-block h-1.5 w-1.5 rounded-full bg-signal" />
              tracking {data?.views.reduce((n, v) => n + v.models.length, 0) ?? 0} model entries
            </span>
            {generatedAt && <span>built {generatedAt}</span>}
          </div>
        </div>
      </header>

      {loading && (
        <div className="cursor-blink p-10 text-center text-ink-dim tracking-widest uppercase text-xs">
          acquiring signal&nbsp;
        </div>
      )}
      {error && (
        <div className="p-10 text-center text-worst text-sm">
          ▲ signal lost — {error}
        </div>
      )}
      {data && (
        <>
          <ViewSelector
            views={data.views}
            selectedId={selectedView?.id ?? data.views[0]?.id ?? ''}
            onSelect={handleSelectView}
          />
          {selectedView && (
            <>
              <VariantBar
                variants={selectedView.variants}
                selectedVariant={selectedVariant || selectedView.variants[0]}
                viewOptions={viewOptions}
                resultCount={rows.length}
                onVariantChange={handleVariantChange}
                onOptionsChange={setViewOptions}
                onExportCsv={handleExportCsv}
                onRefresh={isLiveServer ? refresh : undefined}
              />
              <div className="flex flex-1 min-h-0">
                <DatasetSelector
                  datasets={selectedView.datasets}
                  selected={selectedDataset}
                  onChange={setSelectedDataset}
                />
                <main className="flex-1 overflow-auto p-5">
                  <TaskTable
                    rows={rows}
                    columnDefs={columnDefs}
                    primaryMetrics={selectedView.primary_metrics}
                    viewOptions={viewOptions}
                    onSortChange={handleSortChange}
                  />
                </main>
              </div>
            </>
          )}
        </>
      )}

      {/* Footer strip */}
      <footer className="border-t border-line bg-surface-panel px-6 py-2 text-xs text-ink-dim flex justify-between">
        <span>
          ▲ best per column · cell color = % vs best (green 100% → red 0%) · lower&nbsp;↓ or higher&nbsp;↑ is better
        </span>
        <a
          href="https://github.com/wayne155/pytorch_timeseries"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-phosphor transition-colors"
        >
          github / pytorch_timeseries
        </a>
      </footer>
    </div>
  )
}
