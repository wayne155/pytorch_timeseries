// webapp/src/hooks/useTaskView.ts
import { useMemo } from 'react'
import type { ViewData, TaskTableRow, TaskViewOptions, SubcolumnMetrics } from '../types'

const LOWER_BETTER = ['mse', 'mae', 'loss', 'error', 'rmse', 'crps', 'wis', 'nll', 'mase', 'mape', 'smape']

export function isLowerBetter(key: string): boolean {
  const k = key.toLowerCase()
  return LOWER_BETTER.some(p => k.includes(p))
}

function meanOfSubcolMetrics(subcols: SubcolumnMetrics[]): SubcolumnMetrics {
  if (subcols.length === 0) return {}
  const keys = [...new Set(subcols.flatMap(s => Object.keys(s)))]
  const out: SubcolumnMetrics = {}
  for (const k of keys) {
    const present = subcols.filter(s => k in s)
    if (!present.length) continue
    const meanVal = present.reduce((s, p) => s + p[k].mean, 0) / present.length
    const stdVal = Math.sqrt(present.reduce((s, p) => s + p[k].std ** 2, 0) / present.length)
    const nSeeds = Math.max(...present.map(p => p[k].n_seeds))
    out[k] = { mean: meanVal, std: stdVal, n_seeds: nSeeds }
  }
  return out
}

export interface ColumnDef {
  id: string
  label: string
}

export interface TaskViewResult {
  rows: TaskTableRow[]
  columnDefs: ColumnDef[]
}

export function useTaskView(
  view: ViewData | null,
  variant: string,
  selectedDataset: string,
  options: TaskViewOptions,
): TaskViewResult {
  return useMemo(() => {
    if (!view) return { rows: [], columnDefs: [] }

    const hasSubcolumns = view.subcolumns.length > 0

    // Build column defs
    let columnDefs: ColumnDef[]
    if (hasSubcolumns) {
      columnDefs = [
        { id: 'avg', label: 'avg' },
        ...view.subcolumns.map(sc => ({ id: sc, label: sc })),
      ]
    } else if (selectedDataset === 'All') {
      columnDefs = [
        { id: 'avg', label: 'avg' },
        ...view.datasets.map(d => ({ id: d, label: d })),
      ]
    } else {
      columnDefs = [{ id: 'avg', label: 'avg' }]
    }

    // Build rows
    const complete = new Set<string>()
    const allRows: TaskTableRow[] = view.models.map(model => {
      const variantResults = model.results[variant] ?? {}
      const columns: Record<string, SubcolumnMetrics | null> = {}

      // A model is "complete" when it has results for every dataset of the view.
      if (view.datasets.every(d => {
        const r = variantResults[d]
        return r != null && Object.values(r).some(x => x != null)
      })) {
        complete.add(model.name)
      }

      if (hasSubcolumns) {
        const datasetsToUse = selectedDataset === 'All' ? view.datasets : [selectedDataset]

        for (const sc of view.subcolumns) {
          const subcolAggs = datasetsToUse
            .map(d => variantResults[d]?.[sc])
            .filter((x): x is SubcolumnMetrics => x != null)
          columns[sc] = subcolAggs.length > 0 ? meanOfSubcolMetrics(subcolAggs) : null
        }
        // overall avg = mean of subcolumn avgs; fall back to stored avg if no subcolumn data
        const subcolValues = view.subcolumns
          .map(sc => columns[sc])
          .filter((x): x is SubcolumnMetrics => x != null)
        if (subcolValues.length > 0) {
          columns['avg'] = meanOfSubcolMetrics(subcolValues)
        } else {
          // Fall back to stored avg entries when subcolumn data is absent
          const storedAvgs = datasetsToUse
            .map(d => variantResults[d]?.avg)
            .filter((x): x is SubcolumnMetrics => x != null)
          columns['avg'] = storedAvgs.length > 0 ? meanOfSubcolMetrics(storedAvgs) : null
        }

      } else if (selectedDataset === 'All') {
        const avgInputs: SubcolumnMetrics[] = []
        for (const d of view.datasets) {
          const val = variantResults[d]?.avg ?? null
          columns[d] = val
          if (val) avgInputs.push(val)
        }
        columns['avg'] = avgInputs.length > 0 ? meanOfSubcolMetrics(avgInputs) : null
      } else {
        columns['avg'] = variantResults[selectedDataset]?.avg ?? null
      }

      return {
        model: model.name,
        source_type: model.source_type,
        citation: model.citation,
        url: model.url,
        columns,
      }
    })

    // In the All view, only models with results on every dataset are shown.
    const rows = selectedDataset === 'All'
      ? allRows.filter(r => complete.has(r.model))
      : allRows

    // Sort — defaults to the avg column on the first primary metric, with
    // direction chosen so the best value is on top. Rows missing the sort
    // value always sink to the bottom, whatever the direction.
    const col = options.sortColumn ?? 'avg'
    const metric = options.sortMetric ?? view.primary_metrics[0]
    const dir = options.sortColumn
      ? (options.sortDirection === 'asc' ? 1 : -1)
      : (isLowerBetter(metric ?? '') ? 1 : -1)
    let sorted = rows
    if (metric) {
      sorted = [...rows].sort((a, b) => {
        const av = a.columns[col]?.[metric]?.mean ?? null
        const bv = b.columns[col]?.[metric]?.mean ?? null
        if (av == null && bv == null) return 0
        if (av == null) return 1
        if (bv == null) return -1
        return dir * (av - bv)
      })
    }

    return { rows: sorted, columnDefs }
  }, [view, variant, selectedDataset, options])
}
