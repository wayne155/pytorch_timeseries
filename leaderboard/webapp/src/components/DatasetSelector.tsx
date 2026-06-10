interface DatasetSelectorProps {
  datasets: string[]
  selected: string
  onChange: (dataset: string) => void
}

export function DatasetSelector({ datasets, selected, onChange }: DatasetSelectorProps) {
  const options = ['All', ...datasets]
  return (
    <aside className="w-44 shrink-0 border-r border-line bg-surface-panel overflow-y-auto">
      <div className="px-4 pt-4 pb-2 text-[11px] uppercase tracking-[0.25em] text-ink-faint">
        Datasets
      </div>
      <div className="pb-4">
        {options.map(opt => {
          const active = selected === opt
          return (
            <label
              key={opt}
              className={`flex items-center gap-2.5 px-4 py-1.5 text-[13px] cursor-pointer transition-colors ${
                active
                  ? 'bg-phosphor-dim text-phosphor-bright font-semibold shadow-[inset_3px_0_0_0_#c2780c]'
                  : 'text-ink-dim hover:text-ink hover:bg-surface-raised'
              }`}
            >
              <input
                type="radio"
                name="dataset-selector"
                checked={active}
                onChange={() => onChange(opt)}
                className="instrument-radio"
              />
              <span className="truncate">{opt}</span>
            </label>
          )
        })}
      </div>
    </aside>
  )
}
