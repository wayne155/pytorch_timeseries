import { useState } from 'react'
import type { Filters, LeaderboardSchema } from '../types'

interface FilterSidebarProps {
  schema: LeaderboardSchema
  filters: Filters
  onChange: (f: Filters) => void
}

function SearchCheckboxList({
  label, options, selected, onToggle,
}: { label: string; options: string[]; selected: string[]; onToggle: (v: string) => void }) {
  const [q, setQ] = useState('')
  const visible = q ? options.filter(o => o.toLowerCase().includes(q.toLowerCase())) : options
  return (
    <div className="mb-4">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</div>
      <input
        className="w-full text-xs border border-gray-200 rounded px-2 py-1 mb-1 focus:outline-none"
        placeholder="Search…"
        value={q}
        onChange={e => setQ(e.target.value)}
      />
      <div className="max-h-40 overflow-y-auto space-y-0.5">
        {visible.map(opt => (
          <label key={opt} className="flex items-center gap-1.5 cursor-pointer text-sm hover:bg-gray-50 px-1 rounded">
            <input
              type="checkbox"
              checked={selected.includes(opt)}
              onChange={() => onToggle(opt)}
              className="rounded"
            />
            <span className="truncate">{opt}</span>
          </label>
        ))}
        {visible.length === 0 && <div className="text-xs text-gray-400 px-1">No matches</div>}
      </div>
    </div>
  )
}

export function FilterSidebar({ schema, filters, onChange }: FilterSidebarProps) {
  const toggleList = (key: keyof Pick<Filters, 'datasets' | 'models'>, val: string) => {
    const cur = filters[key]
    onChange({
      ...filters,
      [key]: cur.includes(val) ? cur.filter(v => v !== val) : [...cur, val],
    })
  }

  const setTask = (task: string) => {
    onChange({ task, datasets: [], models: [], hparams: {} })
  }

  const setHparam = (key: string, val: string) => {
    onChange({ ...filters, hparams: { ...filters.hparams, [key]: val === '' ? null : val } })
  }

  const reset = () => onChange({ task: null, datasets: [], models: [], hparams: {} })

  const currentDatasets = filters.task ? (schema.datasets_by_task[filters.task] ?? []) : []
  const currentHparamKeys = filters.task ? (schema.hparams_by_task[filters.task] ?? []) : []

  return (
    <aside className="w-56 shrink-0 bg-white border-r border-gray-200 p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <span className="font-semibold text-gray-800">Filters</span>
        <button onClick={reset} className="text-xs text-blue-500 hover:underline">Reset all</button>
      </div>

      {/* Task */}
      <div className="mb-4">
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Task</div>
        <div className="space-y-0.5">
          <label className="flex items-center gap-1.5 text-sm cursor-pointer hover:bg-gray-50 px-1 rounded">
            <input type="radio" name="task" checked={filters.task === null} onChange={() => reset()} />
            <span>All</span>
          </label>
          {schema.tasks.map(t => (
            <label key={t} className="flex items-center gap-1.5 text-sm cursor-pointer hover:bg-gray-50 px-1 rounded">
              <input type="radio" name="task" checked={filters.task === t} onChange={() => setTask(t)} />
              <span>{t}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Dataset */}
      {currentDatasets.length > 0 && (
        <SearchCheckboxList
          label="Dataset"
          options={currentDatasets}
          selected={filters.datasets}
          onToggle={v => toggleList('datasets', v)}
        />
      )}

      {/* Model */}
      <SearchCheckboxList
        label="Model"
        options={schema.models}
        selected={filters.models}
        onToggle={v => toggleList('models', v)}
      />

      {/* Hparams */}
      {currentHparamKeys.length > 0 && (
        <div className="mb-4">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Hparams</div>
          {currentHparamKeys.map(key => {
            const opts = schema.hparam_options[filters.task!]?.[key] ?? []
            return (
              <div key={key} className="mb-2">
                <div className="text-xs text-gray-500 mb-0.5">{key}</div>
                <select
                  className="w-full text-sm border border-gray-200 rounded px-1.5 py-1 focus:outline-none"
                  value={filters.hparams[key] ?? ''}
                  onChange={e => setHparam(key, e.target.value)}
                >
                  <option value="">All</option>
                  {opts.map(v => (
                    <option key={String(v)} value={String(v)}>{String(v)}</option>
                  ))}
                </select>
              </div>
            )
          })}
        </div>
      )}
    </aside>
  )
}
