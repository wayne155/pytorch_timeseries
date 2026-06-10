const ABBREVS: Record<string, string> = {
  windows: 'w', pred_len: 'p', horizon: 'h',
  mask_rate: 'm', seq_len: 's',
}

interface HparamBadgesProps {
  hparams: Record<string, number | string | boolean>
  onBadgeClick?: (key: string, value: string) => void
}

export function HparamBadges({ hparams, onBadgeClick }: HparamBadgesProps) {
  const pairs = Object.entries(hparams)
  if (pairs.length === 0) return null
  return (
    <div className="flex gap-1 flex-wrap">
      {pairs.map(([k, v]) => (
        <span
          key={k}
          title={`${k}=${v}`}
          className="px-1.5 py-0.5 bg-gray-100 text-gray-600 text-xs rounded cursor-pointer hover:bg-blue-100 select-none"
          onClick={() => onBadgeClick?.(k, String(v))}
        >
          {ABBREVS[k] ?? k[0]}{v}
        </span>
      ))}
    </div>
  )
}
