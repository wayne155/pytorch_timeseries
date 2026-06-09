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

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-3">
        <h1 className="text-lg font-semibold text-gray-900">torch-timeseries Leaderboard</h1>
      </header>

      {loading && <div className="p-8 text-center text-gray-500">Loading...</div>}
      {error && <div className="p-8 text-center text-red-500">Error: {error}</div>}
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
                <main className="flex-1 overflow-auto p-4">
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
    </div>
  )
}
