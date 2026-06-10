import type { AggregatedMetric } from '../types'

interface MetricCellProps {
  value: number | AggregatedMetric
  isBest: boolean
  isWorst: boolean
  showStd: boolean
  /** 1 = column best, 0 = column worst; drives the green→red gradient. */
  score?: number | null
}

export function MetricCell({ value, isBest, isWorst, showStd, score }: MetricCellProps) {
  const numVal = typeof value === 'number' ? value : value.mean
  const std = typeof value === 'object' ? value.std : null

  const cls = isBest ? 'metric-best' : isWorst ? 'metric-worst' : ''

  const pct = score != null && !Number.isNaN(score) ? Math.round(score * 100) : null
  // hue 8° (light red) → 140° (light green), pastel so the number stays legible
  const bg = pct != null ? `hsl(${8 + (pct / 100) * 132}, 60%, 89%)` : undefined

  return (
    <span
      className={`tabular-nums px-1.5 py-0.5 rounded-sm ${cls}`}
      style={bg ? { backgroundColor: bg } : undefined}
      title={pct != null ? `${pct}% (100% = column best)` : undefined}
    >
      {isBest && <span className="mr-1 text-[9px] align-middle">▲</span>}
      {numVal.toFixed(4)}
      {showStd && std !== null && (
        <span className="text-ink-faint text-[11px]"> ±{std.toFixed(4)}</span>
      )}
    </span>
  )
}
