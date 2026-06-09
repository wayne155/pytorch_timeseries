import { useState, useMemo } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { MetricCell } from './MetricCell'
import { isLowerBetter } from '../hooks/useTaskView'
import type { TaskTableRow, TaskViewOptions } from '../types'
import type { ColumnDef as ColDef } from '../hooks/useTaskView'

interface TaskTableProps {
  rows: TaskTableRow[]
  columnDefs: ColDef[]
  primaryMetrics: string[]
  viewOptions: TaskViewOptions
  onSortChange: (column: string, metric: string, dir: 'asc' | 'desc') => void
}

function getBestWorst(
  rows: TaskTableRow[],
  colId: string,
  metric: string,
): { best: number; worst: number } {
  const vals = rows
    .map(r => r.columns[colId]?.[metric]?.mean)
    .filter((v): v is number => v != null)
  if (!vals.length) return { best: NaN, worst: NaN }
  return isLowerBetter(metric)
    ? { best: Math.min(...vals), worst: Math.max(...vals) }
    : { best: Math.max(...vals), worst: Math.min(...vals) }
}

export function TaskTable({
  rows, columnDefs, primaryMetrics, viewOptions, onSortChange,
}: TaskTableProps) {
  const [sorting, setSorting] = useState<SortingState>([])

  const bestWorst = useMemo(() => {
    const bw: Record<string, Record<string, { best: number; worst: number }>> = {}
    for (const col of columnDefs) {
      bw[col.id] = {}
      for (const metric of primaryMetrics) {
        bw[col.id][metric] = getBestWorst(rows, col.id, metric)
      }
    }
    return bw
  }, [rows, columnDefs, primaryMetrics])

  const columns = useMemo<ColumnDef<TaskTableRow>[]>(() => {
    const modelCol: ColumnDef<TaskTableRow> = {
      id: 'model',
      header: 'Model',
      enableSorting: false,
      cell: info => {
        const r = info.row.original
        return r.url
          ? <a href={r.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{r.model}</a>
          : <span>{r.model}</span>
      },
    }

    const metricGroups: ColumnDef<TaskTableRow>[] = columnDefs.map(col => ({
      id: col.id,
      header: col.label === 'avg'
        ? () => <span className="font-semibold">avg</span>
        : col.label,
      enableSorting: false,
      columns: primaryMetrics.map((metric): ColumnDef<TaskTableRow> => ({
        id: `${col.id}__${metric}`,
        header: () => (
          <span
            className="cursor-pointer select-none hover:text-blue-600 flex items-center gap-0.5"
            title={isLowerBetter(metric) ? 'lower is better' : 'higher is better'}
          >
            {metric} {isLowerBetter(metric) ? '↓' : '↑'}
          </span>
        ),
        accessorFn: (row: TaskTableRow) => row.columns[col.id]?.[metric]?.mean ?? null,
        cell: info => {
          const cellData = info.row.original.columns[col.id]
          if (!cellData || !(metric in cellData)) {
            return <span className="text-gray-300 tabular-nums">—</span>
          }
          const m = cellData[metric]
          const { best, worst } = bestWorst[col.id]?.[metric] ?? { best: NaN, worst: NaN }
          return (
            <MetricCell
              value={{ mean: m.mean, std: m.std }}
              isBest={m.mean === best}
              isWorst={m.mean === worst}
              showStd={viewOptions.showStd}
            />
          )
        },
      })),
    }))

    const actionCol: ColumnDef<TaskTableRow> = {
      id: '_action',
      header: '',
      enableSorting: false,
      size: 40,
      cell: () => (
        <button
          title="Row actions"
          disabled
          className="text-gray-300 px-1.5 py-0.5 rounded text-base leading-none cursor-not-allowed"
        >
          ⋯
        </button>
      ),
    }

    return [modelCol, ...metricGroups, actionCol]
  }, [columnDefs, primaryMetrics, bestWorst, viewOptions.showStd])

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting },
    onSortingChange: updater => {
      const next = typeof updater === 'function' ? updater(sorting) : updater
      setSorting(next)
      if (next.length > 0) {
        const parts = next[0].id.split('__')
        if (parts.length === 2) {
          onSortChange(parts[0], parts[1], next[0].desc ? 'desc' : 'asc')
        }
      }
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm text-left border-collapse">
        <thead className="bg-gray-50 border-b border-gray-200 sticky top-0 z-10">
          {table.getHeaderGroups().map(hg => (
            <tr key={hg.id}>
              {hg.headers.map(header => (
                <th
                  key={header.id}
                  colSpan={header.colSpan}
                  style={{ width: header.getSize() }}
                  className={`px-3 py-1.5 text-xs font-semibold text-gray-600 whitespace-nowrap border-r border-gray-100 last:border-r-0 ${
                    header.column.getCanSort() ? 'cursor-pointer select-none hover:bg-gray-100' : ''
                  } ${header.depth === 0 && header.colSpan > 1 ? 'text-center border-b border-gray-200' : ''}`}
                  onClick={header.column.getToggleSortingHandler()}
                >
                  {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="divide-y divide-gray-100">
          {table.getRowModel().rows.map((row, i) => (
            <tr key={row.id} className={`hover:bg-gray-50 ${i % 2 === 0 ? '' : 'bg-gray-50/30'}`}>
              {row.getVisibleCells().map(cell => (
                <td key={cell.id} className="px-3 py-1.5 whitespace-nowrap border-r border-gray-50 last:border-r-0">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td
                colSpan={99}
                className="px-3 py-8 text-center text-gray-400"
              >
                No results for this view/variant/dataset combination.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
