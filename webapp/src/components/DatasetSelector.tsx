interface DatasetSelectorProps {
  datasets: string[]
  selected: string
  onChange: (dataset: string) => void
}

export function DatasetSelector({ datasets, selected, onChange }: DatasetSelectorProps) {
  const options = ['All', ...datasets]
  return (
    <aside className="w-44 shrink-0 bg-white border-r border-gray-200 overflow-y-auto">
      <div className="px-3 pt-3 pb-1 text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Datasets
      </div>
      <div className="pb-3">
        {options.map(opt => (
          <label
            key={opt}
            className="flex items-center gap-2 px-3 py-1 text-sm cursor-pointer hover:bg-gray-50"
          >
            <input
              type="radio"
              name="dataset-selector"
              checked={selected === opt}
              onChange={() => onChange(opt)}
              className="accent-blue-600"
            />
            <span className={`truncate ${selected === opt ? 'font-medium text-gray-900' : 'text-gray-700'}`}>
              {opt}
            </span>
          </label>
        ))}
      </div>
    </aside>
  )
}
