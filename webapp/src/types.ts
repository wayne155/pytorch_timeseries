export interface Entry {
  id: string
  model: string
  task: string
  dataset: string
  seed: number | null
  hparams: Record<string, number | string | boolean>
  metrics: Record<string, number>
  num_params: number | null
  train_time_sec: number | null
  git_commit: string
  timestamp: string
  source_type: 'local_run' | 'paper'
  citation: string
  url: string
  notes: string
}

export interface AggregatedMetric {
  mean: number
  std: number
}

/** One row in the table — either a per-seed Entry or an aggregated group. */
export interface DisplayRow {
  key: string
  model: string
  task: string
  dataset: string
  hparams: Record<string, number | string | boolean>
  metrics: Record<string, number | AggregatedMetric>
  num_seeds: number
  seed: number | null
  source_type: string
  citation: string
  url: string
  isAggregated: boolean
}

export interface LeaderboardSchema {
  tasks: string[]
  datasets_by_task: Record<string, string[]>
  models: string[]
  hparams_by_task: Record<string, string[]>
  hparam_options: Record<string, Record<string, (number | string)[]>>
}

export interface LeaderboardData {
  generated_at: string
  entries: Entry[]
  schema: LeaderboardSchema
}

export interface Filters {
  task: string | null
  datasets: string[]
  models: string[]
  hparams: Record<string, string | null>
}

export interface ViewOptions {
  aggregate: boolean
  showStd: boolean
  visibleMetrics: string[]
  sortColumn: string | null
  sortDirection: 'asc' | 'desc'
}
