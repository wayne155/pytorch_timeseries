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

  const cls = isBest
    ? 'bg-green-50 text-green-800 font-semibold'
    : isWorst
    ? 'bg-red-50 text-red-400'
    : ''

  return (
    <span className={`tabular-nums px-1 rounded ${cls}`}>
      {numVal.toFixed(4)}
      {showStd && std !== null && (
        <span className="text-gray-400 text-xs"> ±{std.toFixed(4)}</span>
      )}
    </span>
  )
}
