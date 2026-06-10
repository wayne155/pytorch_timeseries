// webapp/src/hooks/useTaskView.test.ts
import { describe, it, expect } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useTaskView } from './useTaskView'
import type { ViewData, TaskViewOptions } from '../types'

const DEFAULT_OPTS: TaskViewOptions = {
  showStd: false, sortColumn: null, sortMetric: null, sortDirection: 'asc',
}

function makeView(overrides: Partial<ViewData> = {}): ViewData {
  return {
    id: 'long_term_forecast',
    display_name: 'Long-Term Forecast',
    primary_metrics: ['mse', 'mae'],
    variants: ['I96'],
    datasets: ['ETTh1', 'ETTh2'],
    subcolumns: ['96', '192'],
    models: [
      {
        name: 'DLinear',
        source_type: 'local_run',
        citation: '',
        url: '',
        results: {
          I96: {
            ETTh1: {
              avg: { mse: { mean: 0.417, std: 0.008, n_seeds: 3 }, mae: { mean: 0.300, std: 0.004, n_seeds: 3 } },
              '96':  { mse: { mean: 0.384, std: 0.007, n_seeds: 3 }, mae: { mean: 0.276, std: 0.003, n_seeds: 3 } },
              '192': { mse: { mean: 0.450, std: 0.009, n_seeds: 3 }, mae: { mean: 0.324, std: 0.005, n_seeds: 3 } },
            },
            ETTh2: {
              avg: { mse: { mean: 0.380, std: 0.005, n_seeds: 3 }, mae: { mean: 0.290, std: 0.003, n_seeds: 3 } },
              '96':  { mse: { mean: 0.350, std: 0.005, n_seeds: 3 }, mae: { mean: 0.265, std: 0.002, n_seeds: 3 } },
              '192': { mse: { mean: 0.410, std: 0.006, n_seeds: 3 }, mae: { mean: 0.315, std: 0.004, n_seeds: 3 } },
            },
          },
        },
      },
    ],
    ...overrides,
  }
}

describe('useTaskView — with subcolumns (Forecast)', () => {
  it('returns one row per model when dataset is specific', () => {
    const view = makeView()
    const { result } = renderHook(() =>
      useTaskView(view, 'I96', 'ETTh1', DEFAULT_OPTS)
    )
    expect(result.current.rows).toHaveLength(1)
    const row = result.current.rows[0]
    expect(row.model).toBe('DLinear')
    expect(row.columns['avg']?.['mse'].mean).toBeCloseTo(0.417)
    expect(row.columns['96']?.['mse'].mean).toBeCloseTo(0.384)
    expect(row.columns['192']?.['mse'].mean).toBeCloseTo(0.450)
  })

  it('returns avg-across-datasets when dataset is All', () => {
    const view = makeView()
    const { result } = renderHook(() =>
      useTaskView(view, 'I96', 'All', DEFAULT_OPTS)
    )
    const row = result.current.rows[0]
    // avg across ETTh1 (0.417) and ETTh2 (0.380) → 0.3985
    expect(row.columns['avg']?.['mse'].mean).toBeCloseTo(0.3985)
    expect(row.columns['96']?.['mse'].mean).toBeCloseTo((0.384 + 0.350) / 2)
  })

  it('columnDefs includes avg + subcolumns when dataset is specific', () => {
    const view = makeView()
    const { result } = renderHook(() =>
      useTaskView(view, 'I96', 'ETTh1', DEFAULT_OPTS)
    )
    const ids = result.current.columnDefs.map(c => c.id)
    expect(ids).toContain('avg')
    expect(ids).toContain('96')
    expect(ids).toContain('192')
  })
})

describe('useTaskView — no subcolumns (UEA)', () => {
  it('returns one column per dataset when All selected', () => {
    const uea = makeView({
      id: 'uea_classification',
      subcolumns: [],
      primary_metrics: ['accuracy'],
      models: [{
        name: 'GRU-D', source_type: 'local_run', citation: '', url: '',
        results: {
          W96: {
            EthanolConcentration: { avg: { accuracy: { mean: 0.71, std: 0.02, n_seeds: 3 } } },
            FaceDetection:        { avg: { accuracy: { mean: 0.75, std: 0.01, n_seeds: 3 } } },
          },
        },
      }],
      variants: ['W96'],
      datasets: ['EthanolConcentration', 'FaceDetection'],
    })
    const { result } = renderHook(() =>
      useTaskView(uea, 'W96', 'All', DEFAULT_OPTS)
    )
    const ids = result.current.columnDefs.map(c => c.id)
    expect(ids).toContain('avg')
    expect(ids).toContain('EthanolConcentration')
    expect(ids).toContain('FaceDetection')
    const row = result.current.rows[0]
    expect(row.columns['EthanolConcentration']?.['accuracy'].mean).toBeCloseTo(0.71)
  })

  it('returns only avg column when specific dataset selected', () => {
    const uea = makeView({
      id: 'uea_classification',
      subcolumns: [],
      primary_metrics: ['accuracy'],
      models: [{
        name: 'GRU-D', source_type: 'local_run', citation: '', url: '',
        results: {
          W96: {
            EthanolConcentration: { avg: { accuracy: { mean: 0.71, std: 0.02, n_seeds: 3 } } },
          },
        },
      }],
      variants: ['W96'],
      datasets: ['EthanolConcentration'],
    })
    const { result } = renderHook(() =>
      useTaskView(uea, 'W96', 'EthanolConcentration', DEFAULT_OPTS)
    )
    const ids = result.current.columnDefs.map(c => c.id)
    expect(ids).toEqual(['avg'])
  })
})

describe('useTaskView — sorting', () => {
  it('sorts rows by avg mse ascending', () => {
    const view = makeView({
      models: [
        { name: 'Worse', source_type: 'local_run', citation: '', url: '',
          results: { I96: { ETTh1: { avg: { mse: { mean: 0.50, std: 0.01, n_seeds: 3 } } } } } },
        { name: 'Better', source_type: 'local_run', citation: '', url: '',
          results: { I96: { ETTh1: { avg: { mse: { mean: 0.38, std: 0.01, n_seeds: 3 } } } } } },
      ],
    })
    const opts = { ...DEFAULT_OPTS, sortColumn: 'avg', sortMetric: 'mse', sortDirection: 'asc' as const }
    const { result } = renderHook(() => useTaskView(view, 'I96', 'ETTh1', opts))
    expect(result.current.rows[0].model).toBe('Better')
    expect(result.current.rows[1].model).toBe('Worse')
  })
})
