// webapp/src/types.ts

export interface SubcolumnMetrics {
  [metric: string]: { mean: number; std: number; n_seeds: number }
}

export interface DatasetResult {
  avg: SubcolumnMetrics
  [subcolumn: string]: SubcolumnMetrics
}

export interface ModelResult {
  name: string
  source_type: 'local_run' | 'paper'
  citation: string
  url: string
  results: {
    [variant: string]: {
      [dataset: string]: DatasetResult
    }
  }
}

export interface ViewData {
  id: string
  display_name: string
  primary_metrics: string[]
  variants: string[]
  datasets: string[]
  subcolumns: string[]
  models: ModelResult[]
}

export interface TaskLeaderboardData {
  generated_at: string
  views: ViewData[]
  schema: {
    views: string[]
    datasets_by_view: Record<string, string[]>
  }
}

/** One display row in TaskTable. */
export interface TaskTableRow {
  model: string
  source_type: string
  citation: string
  url: string
  /** colId → aggregated metrics. colId is "avg" | subcolumn label | dataset name */
  columns: Record<string, SubcolumnMetrics | null>
}

export interface TaskViewOptions {
  showStd: boolean
  sortColumn: string | null
  sortMetric: string | null
  sortDirection: 'asc' | 'desc'
}
