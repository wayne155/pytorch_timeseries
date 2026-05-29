import { useMemo, useState } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { MetricCell } from './MetricCell'
import { HparamBadges } from './HparamBadges'
import { getBestWorst, isLowerBetter } from '../hooks/useFiltered'
import type { DisplayRow, ViewOptions } from '../types'

interface LeaderboardTableProps {
  rows: DisplayRow[]
  viewOptions: ViewOptions
  onSortChange: (col: string, dir: 'asc' | 'desc') => void
  onHparamClick: (key: string, value: string) => void
}

export function LeaderboardTable({
  rows, viewOptions, onSortChange, onHparamClick,
}: LeaderboardTableProps) {
  const [sorting, setSorting] = useState<SortingState>([])

  const visibleMetrics = viewOptions.visibleMetrics

  const bestWorstByMetric = useMemo(() => {
    const result: Record<string, { best: number; worst: number }> = {}
    for (const key of visibleMetrics) {
      result[key] = getBestWorst(rows, key)
    }
    return result
  }, [rows, visibleMetrics])

  const columns = useMemo<ColumnDef<DisplayRow>[]>(() => {
    const staticCols: ColumnDef<DisplayRow>[] = [
      {
        id: 'rank',
        header: '#',
        cell: info => info.row.index + 1,
        enableSorting: false,
        size: 48,
      },
      {
        accessorKey: 'model',
        header: 'Model',
        cell: info => {
          const row = info.row.original
          return row.url
            ? <a href={row.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{row.model}</a>
            : row.model
        },
      },
      {
        accessorKey: 'dataset',
        header: 'Dataset',
      },
      {
        id: 'hparams',
        header: 'Config',
        cell: info => (
          <HparamBadges hparams={info.row.original.hparams} onBadgeClick={onHparamClick} />
        ),
        enableSorting: false,
      },
      {
        id: 'seeds',
        header: 'Seeds',
        cell: info => {
          const r = info.row.original
          return r.isAggregated ? `n=${r.num_seeds}` : `#${r.seed ?? '—'}`
        },
        enableSorting: false,
        size: 64,
      },
      {
        id: 'source',
        header: 'Source',
        cell: info => {
          const r = info.row.original
          const isPaper = r.source_type === 'paper'
          return (
            <span
              title={r.citation || undefined}
              className={`text-xs px-1.5 py-0.5 rounded ${isPaper ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'}`}
            >
              {isPaper ? 'paper' : 'local'}
            </span>
          )
        },
        enableSorting: false,
        size: 72,
      },
    ]

    const metricCols: ColumnDef<DisplayRow>[] = visibleMetrics.map(key => ({
      id: key,
      header: () => (
        <span title={isLowerBetter(key) ? 'lower is better' : 'higher is better'}>
          {key} {isLowerBetter(key) ? '↓' : '↑'}
        </span>
      ),
      accessorFn: (row: DisplayRow) => {
        const m = row.metrics[key]
        return m === undefined ? null : typeof m === 'number' ? m : m.mean
      },
      cell: info => {
        const val = info.row.original.metrics[key]
        if (val === undefined) return <span className="text-gray-300">—</span>
        const mean = typeof val === 'number' ? val : val.mean
        const { best, worst } = bestWorstByMetric[key] ?? { best: NaN, worst: NaN }
        return (
          <MetricCell
            value={val}
            isBest={mean === best}
            isWorst={mean === worst}
            showStd={viewOptions.showStd && info.row.original.isAggregated}
          />
        )
      },
    }))

    return [...staticCols, ...metricCols]
  }, [visibleMetrics, bestWorstByMetric, viewOptions.showStd, onHparamClick])

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting },
    onSortingChange: (updater) => {
      const next = typeof updater === 'function' ? updater(sorting) : updater
      setSorting(next)
      if (next.length > 0) onSortChange(next[0].id, next[0].desc ? 'desc' : 'asc')
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    manualSorting: false,
  })

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm text-left">
        <thead className="bg-gray-50 border-b border-gray-200 sticky top-0">
          {table.getHeaderGroups().map(hg => (
            <tr key={hg.id}>
              {hg.headers.map(header => (
                <th
                  key={header.id}
                  style={{ width: header.getSize() }}
                  className={`px-3 py-2 text-xs font-semibold text-gray-600 whitespace-nowrap ${header.column.getCanSort() ? 'cursor-pointer select-none hover:bg-gray-100' : ''}`}
                  onClick={header.column.getToggleSortingHandler()}
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                  {header.column.getIsSorted() === 'asc' ? ' ▲' : header.column.getIsSorted() === 'desc' ? ' ▼' : ''}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="divide-y divide-gray-100">
          {table.getRowModel().rows.map(row => (
            <tr key={row.id} className="hover:bg-gray-50">
              {row.getVisibleCells().map(cell => (
                <td key={cell.id} className="px-3 py-2 whitespace-nowrap">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-3 py-8 text-center text-gray-400">
                No results match the current filters.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
