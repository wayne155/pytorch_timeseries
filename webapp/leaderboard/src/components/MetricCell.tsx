import type { AggregatedMetric } from '../types'

interface MetricCellProps {
  value: number | AggregatedMetric
  isBest: boolean
  isWorst: boolean
  showStd: boolean
  /** Fraction of the column best (1 = best, <1 = worse); drives the data bar. */
  score?: number | null
}

export function MetricCell({ value, isBest, isWorst, showStd, score }: MetricCellProps) {
  const numVal = typeof value === 'number' ? value : value.mean
  const std = typeof value === 'object' ? value.std : null

  const cls = isBest ? 'metric-best' : isWorst ? 'metric-worst' : ''

  const pct = score != null && !Number.isNaN(score)
    ? Math.round(Math.max(0, Math.min(1, score)) * 100)
    : null
  // Excel-style data bar: cell background fills left→right up to pct,
  // red (hue 8°) at 0% → green (hue 140°) at 100%, pastel so text stays legible.
  const fill = pct != null
    ? `linear-gradient(to right, hsl(${8 + (pct / 100) * 132}, 55%, 84%) ${pct}%, transparent ${pct}%)`
    : undefined

  return (
    <span
      className={`inline-flex w-full min-w-[88px] items-baseline gap-1 tabular-nums px-1.5 py-0.5 rounded-sm ${cls}`}
      style={fill ? { backgroundImage: fill } : undefined}
      title={pct != null ? `${pct}% of column best` : undefined}
    >
      {isBest && <span className="text-[9px] self-center">▲</span>}
      {numVal.toFixed(4)}
      {showStd && std !== null && (
        <span className="text-ink-faint text-[11px]">±{std.toFixed(4)}</span>
      )}
      {pct != null && (
        <span className="ml-auto text-[9px] text-ink-faint">{pct}%</span>
      )}
    </span>
  )
}
