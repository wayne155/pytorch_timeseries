import type { AggregatedMetric } from '../types'

interface MetricCellProps {
  value: number | AggregatedMetric
  isBest: boolean
  isWorst: boolean
  showStd: boolean
}

export function MetricCell({ value, isBest, isWorst, showStd }: MetricCellProps) {
  const numVal = typeof value === 'number' ? value : value.mean
  const std = typeof value === 'object' ? value.std : null

  const cls = isBest ? 'metric-best' : isWorst ? 'metric-worst' : ''

  return (
    <span className={`tabular-nums px-1.5 py-0.5 rounded-sm ${cls}`}>
      {isBest && <span className="mr-1 text-[9px] align-middle">▲</span>}
      {numVal.toFixed(4)}
      {showStd && std !== null && (
        <span className="text-ink-faint text-[11px]"> ±{std.toFixed(4)}</span>
      )}
    </span>
  )
}
