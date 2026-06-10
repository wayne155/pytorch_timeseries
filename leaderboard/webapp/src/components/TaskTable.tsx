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
  if (vals.length <= 1) return { best: NaN, worst: NaN }
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
        const idx = String(info.row.index + 1).padStart(2, '0')
        return (
          <span className="flex items-center gap-2.5">
            <span className="text-[11px] text-ink-faint tabular-nums">{idx}</span>
            {r.url ? (
              <a
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-signal hover:text-phosphor-bright hover:underline underline-offset-4 transition-colors"
              >
                {r.model}
              </a>
            ) : (
              <span className="text-ink font-medium">{r.model}</span>
            )}
            {r.source_type === 'paper' && (
              <span
                className="rounded-sm border border-line px-1 py-px text-[9px] uppercase tracking-wider text-ink-faint"
                title={r.citation || 'curated paper result'}
              >
                paper
              </span>
            )}
          </span>
        )
      },
    }

    const metricGroups: ColumnDef<TaskTableRow>[] = columnDefs.map(col => ({
      id: col.id,
      header: col.label === 'avg'
        ? () => <span className="font-semibold text-phosphor">avg</span>
        : col.label,
      enableSorting: false,
      columns: primaryMetrics.map((metric): ColumnDef<TaskTableRow> => ({
        id: `${col.id}__${metric}`,
        enableSorting: true,
        sortDescFirst: !isLowerBetter(metric),
        header: () => (
          <span
            className="cursor-pointer select-none hover:text-phosphor-bright flex items-center gap-0.5 transition-colors"
            title={isLowerBetter(metric) ? 'lower is better' : 'higher is better'}
          >
            {metric} {isLowerBetter(metric) ? '↓' : '↑'}
          </span>
        ),
        accessorFn: (row: TaskTableRow) => row.columns[col.id]?.[metric]?.mean ?? null,
        cell: info => {
          const cellData = info.row.original.columns[col.id]
          if (!cellData || !(metric in cellData)) {
            return <span className="text-ink-faint tabular-nums">—</span>
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
          className="text-ink-faint/60 px-1.5 py-0.5 rounded text-base leading-none cursor-not-allowed"
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
    <div className="overflow-x-auto rounded border border-line bg-surface-panel shadow-[0_6px_24px_rgba(28,27,23,0.07)]">
      <table className="min-w-full text-[13px] text-left border-collapse">
        <thead className="sticky top-0 z-10 bg-surface-raised/95 backdrop-blur">
          {table.getHeaderGroups().map(hg => (
            <tr key={hg.id}>
              {hg.headers.map(header => (
                <th
                  key={header.id}
                  colSpan={header.colSpan}
                  style={{ width: header.getSize() }}
                  className={`px-3 py-2 text-xs uppercase tracking-wider text-ink-dim whitespace-nowrap border-b border-r border-line last:border-r-0 ${
                    header.column.getCanSort() ? 'cursor-pointer select-none hover:bg-phosphor-dim hover:text-ink' : ''
                  } ${header.depth === 0 && header.colSpan > 1 ? 'text-center' : ''} ${
                    header.column.getIsSorted() ? 'bg-phosphor-dim text-phosphor-bright font-semibold' : ''
                  }`}
                  onClick={header.column.getToggleSortingHandler()}
                >
                  {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row, i) => (
            <tr
              key={row.id}
              className="row-in border-b border-line last:border-b-0 transition-colors hover:bg-surface-raised hover:shadow-[inset_3px_0_0_0_#c2780c]"
              style={{ animationDelay: `${Math.min(i * 35, 420)}ms` }}
            >
              {row.getVisibleCells().map(cell => (
                <td key={cell.id} className="px-3 py-2 whitespace-nowrap border-r border-line/60 last:border-r-0">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td
                colSpan={99}
                className="px-3 py-12 text-center text-ink-faint tracking-wide"
              >
                ∅ &nbsp;No results for this view/variant/dataset combination.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
