import { useMemo } from 'react'
import type { Entry, DisplayRow, Filters, ViewOptions, AggregatedMetric } from '../types'

const LOWER_BETTER_PATTERNS = ['mse', 'mae', 'loss', 'error', 'rmse', 'crps', 'wis', 'nll']

export function isLowerBetter(key: string): boolean {
  const k = key.toLowerCase()
  return LOWER_BETTER_PATTERNS.some(p => k.includes(p))
}

function sortedHparamKey(hparams: Record<string, number | string | boolean>): string {
  return JSON.stringify(Object.fromEntries(Object.entries(hparams).sort()))
}

function groupKey(e: Entry): string {
  return `${e.model}|${e.task}|${e.dataset}|${sortedHparamKey(e.hparams)}`
}

function aggregateGroup(group: Entry[]): DisplayRow {
  const first = group[0]
  const metricKeys = [...new Set(group.flatMap(e => Object.keys(e.metrics)))]
  const metrics: Record<string, AggregatedMetric> = {}
  for (const k of metricKeys) {
    const vals = group.map(e => e.metrics[k]).filter((v): v is number => v !== undefined)
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length
    const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(vals.length - 1, 1)
    metrics[k] = { mean, std: Math.sqrt(variance) }
  }
  return {
    key: groupKey(first),
    model: first.model, task: first.task, dataset: first.dataset,
    hparams: first.hparams, metrics, num_seeds: group.length,
    seed: null, source_type: first.source_type,
    citation: first.citation, url: first.url, isAggregated: true,
  }
}

function entryToRow(e: Entry): DisplayRow {
  return {
    key: e.id, model: e.model, task: e.task, dataset: e.dataset,
    hparams: e.hparams, metrics: e.metrics, num_seeds: 1,
    seed: e.seed, source_type: e.source_type,
    citation: e.citation, url: e.url, isAggregated: false,
  }
}

function applyFilters(entries: Entry[], filters: Filters): Entry[] {
  return entries.filter(e => {
    if (filters.task && e.task !== filters.task) return false
    if (filters.datasets.length && !filters.datasets.includes(e.dataset)) return false
    if (filters.models.length && !filters.models.includes(e.model)) return false
    for (const [key, val] of Object.entries(filters.hparams)) {
      if (val === null) continue
      if (String(e.hparams[key]) !== val) return false
    }
    return true
  })
}

function metricMean(row: DisplayRow, col: string): number | null {
  const m = row.metrics[col]
  if (m === undefined) return null
  return typeof m === 'number' ? m : m.mean
}

export function getBestWorst(rows: DisplayRow[], metricKey: string): { best: number; worst: number } {
  const vals = rows.map(r => metricMean(r, metricKey)).filter((v): v is number => v !== null)
  if (vals.length === 0) return { best: NaN, worst: NaN }
  const asc = isLowerBetter(metricKey)
  return {
    best: asc ? Math.min(...vals) : Math.max(...vals),
    worst: asc ? Math.max(...vals) : Math.min(...vals),
  }
}

export function useFiltered(
  entries: Entry[],
  filters: Filters,
  viewOptions: ViewOptions,
): DisplayRow[] {
  return useMemo(() => {
    const filtered = applyFilters(entries, filters)

    let rows: DisplayRow[]
    if (viewOptions.aggregate) {
      const groups = new Map<string, Entry[]>()
      for (const e of filtered) {
        const k = groupKey(e)
        if (!groups.has(k)) groups.set(k, [])
        groups.get(k)!.push(e)
      }
      rows = Array.from(groups.values()).map(aggregateGroup)
    } else {
      rows = filtered.map(entryToRow)
    }

    if (!viewOptions.sortColumn) return rows
    const col = viewOptions.sortColumn
    const dir = viewOptions.sortDirection === 'asc' ? 1 : -1
    return [...rows].sort((a, b) => {
      const av = col === 'model' ? a.model : (metricMean(a, col) ?? 0)
      const bv = col === 'model' ? b.model : (metricMean(b, col) ?? 0)
      if (typeof av === 'string') return dir * av.localeCompare(bv as string)
      return dir * ((av as number) - (bv as number))
    })
  }, [entries, filters, viewOptions])
}
