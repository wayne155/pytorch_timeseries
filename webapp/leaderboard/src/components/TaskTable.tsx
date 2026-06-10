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
  /** Shown at the top-left of the table; explains what the sub-columns mean. */
  columnsNote?: string
}

/** Paper links for bundled models; curated entries can override via row.url. */
const PAPER_LINKS: Record<string, string> = {
  DLinear: 'https://arxiv.org/abs/2205.13504',
  PatchTST: 'https://arxiv.org/abs/2211.14730',
  iTransformer: 'https://arxiv.org/abs/2310.06625',
  Autoformer: 'https://arxiv.org/abs/2106.13008',
  FEDformer: 'https://arxiv.org/abs/2201.12740',
  Informer: 'https://arxiv.org/abs/2012.07436',
  Crossformer: 'https://openreview.net/forum?id=vSVLM2j9eie',
  FITS: 'https://arxiv.org/abs/2307.03756',
  FreTS: 'https://arxiv.org/abs/2311.06184',
  CATS: 'https://arxiv.org/abs/2403.01673',
  'GRU-D': 'https://arxiv.org/abs/1606.01865',
}

/** Official code repositories published with each paper. */
const OFFICIAL_REPOS: Record<string, string> = {
  DLinear: 'https://github.com/cure-lab/LTSF-Linear',
  PatchTST: 'https://github.com/yuqinie98/PatchTST',
  iTransformer: 'https://github.com/thuml/iTransformer',
  Autoformer: 'https://github.com/thuml/Autoformer',
  FEDformer: 'https://github.com/MAZiqing/FEDformer',
  Informer: 'https://github.com/zhouhaoyi/Informer2020',
  Crossformer: 'https://github.com/Thinklab-SJTU/Crossformer',
  FITS: 'https://github.com/VEWOXIC/FITS',
  FreTS: 'https://github.com/aikunyi/FreTS',
  CATS: 'https://github.com/dongbeank/CATS',
  'GRU-D': 'https://github.com/PeterChe1990/GRU-D',
}

/** Training paradigm per model; anything unlisted defaults to supervised. */
const MODEL_PARADIGMS: Record<string, string> = {
  // e.g. 'Chronos': 'pretrained', 'TimesFM': 'pretrained'
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
    </svg>
  )
}

function PaperIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.3" aria-hidden="true">
      <path d="M4 1.5h5.5L13 5v9.5H4z" />
      <path d="M9.5 1.5V5H13M6 8h4M6 10.5h4" />
    </svg>
  )
}

function getBestWorst(
  rows: TaskTableRow[],
  colId: string,
  metric: string,
): { best: number; worst: number } {
  const vals = rows
    .map(r => r.columns[colId]?.[metric]?.mean)
    .filter((v): v is number => v != null)
  if (vals.length === 0) return { best: NaN, worst: NaN }
  return isLowerBetter(metric)
    ? { best: Math.min(...vals), worst: Math.max(...vals) }
    : { best: Math.max(...vals), worst: Math.min(...vals) }
}

export function TaskTable({
  rows, columnDefs, primaryMetrics, viewOptions, onSortChange, columnsNote,
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
    // gold / silver / bronze
    const MEDAL_COLORS = ['#C9A227', '#9DA4AE', '#B0713D']
    const rankCol: ColumnDef<TaskTableRow> = {
      id: '_rank',
      header: '#',
      enableSorting: false,
      size: 44,
      cell: info => {
        const pos = info.table.getRowModel().rows.findIndex(r => r.id === info.row.id) + 1
        return pos <= 3 ? (
          <span
            className="inline-flex h-5 w-5 items-center justify-center rounded-full text-white text-[11px] font-semibold tabular-nums shadow-sm"
            style={{ backgroundColor: MEDAL_COLORS[pos - 1] }}
            title={['gold', 'silver', 'bronze'][pos - 1]}
          >
            {pos}
          </span>
        ) : (
          <span className="inline-flex h-5 w-5 items-center justify-center text-[12px] text-ink-dim tabular-nums">
            {pos}
          </span>
        )
      },
    }

    const typeCol: ColumnDef<TaskTableRow> = {
      id: '_type',
      header: 'Type',
      enableSorting: false,
      size: 90,
      cell: info => {
        const paradigm = MODEL_PARADIGMS[info.row.original.model] ?? 'supervised'
        return (
          <span
            className={`rounded-sm px-1.5 py-px text-[10px] uppercase tracking-wider ${
              paradigm === 'pretrained'
                ? 'bg-phosphor-dim text-phosphor-bright'
                : 'bg-signal-dim text-signal'
            }`}
          >
            {paradigm}
          </span>
        )
      },
    }

    const modelCol: ColumnDef<TaskTableRow> = {
      id: 'model',
      header: 'Model',
      enableSorting: false,
      cell: info => {
        const r = info.row.original
        const paperUrl = r.url || PAPER_LINKS[r.model]
        const repoUrl = OFFICIAL_REPOS[r.model]
        return (
          <span className="flex items-center gap-2">
            <span className="text-ink font-medium">{r.model}</span>
            {repoUrl && (
              <a
                href={repoUrl}
                target="_blank"
                rel="noopener noreferrer"
                title={`${r.model} official code (from the paper)`}
                className="text-ink-faint hover:text-ink transition-colors"
              >
                <GitHubIcon />
              </a>
            )}
            {paperUrl && (
              <a
                href={paperUrl}
                target="_blank"
                rel="noopener noreferrer"
                title={r.citation || `${r.model} paper`}
                className="text-ink-faint hover:text-phosphor-bright transition-colors"
              >
                <PaperIcon />
              </a>
            )}
            {r.source_type === 'paper' && (
              <span
                className="rounded-sm border border-line px-1 py-px text-[9px] uppercase tracking-wider text-ink-faint"
                title={r.citation || 'numbers copied from the paper, not reproduced'}
              >
                paper #s
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
          // Fraction of column best: 1 for the best entry, <1 the further away.
          const score = Number.isNaN(best) || best === 0 || m.mean === 0
            ? null
            : isLowerBetter(metric) ? best / m.mean : m.mean / best
          const contested = !Number.isNaN(best) && best !== worst
          return (
            <MetricCell
              value={{ mean: m.mean, std: m.std }}
              isBest={contested && m.mean === best}
              isWorst={contested && m.mean === worst}
              showStd={viewOptions.showStd}
              score={score}
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

    return [rankCol, typeCol, modelCol, ...metricGroups, actionCol]
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
      {columnsNote && (
        <div className="border-b border-line bg-surface-raised/70 px-3 py-1.5 text-[11px] text-ink-dim">
          ⓘ {columnsNote}
        </div>
      )}
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
