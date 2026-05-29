import { describe, it, expect } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useFiltered, isLowerBetter, getBestWorst } from './useFiltered'
import type { Entry, Filters, ViewOptions } from '../types'

function makeEntry(overrides: Partial<Entry> = {}): Entry {
  return {
    id: 'id1', model: 'PatchTST', task: 'Forecast', dataset: 'ETTh1',
    seed: 1, hparams: { windows: 96, pred_len: 96, horizon: 1 },
    metrics: { mse: 0.38, mae: 0.28 }, num_params: null, train_time_sec: null,
    git_commit: '', timestamp: '', source_type: 'local_run',
    citation: '', url: '', notes: '', ...overrides,
  }
}

const VIEW_IND: ViewOptions = {
  aggregate: false, showStd: false,
  visibleMetrics: ['mse', 'mae'], sortColumn: null, sortDirection: 'asc',
}
const VIEW_AGG: ViewOptions = { ...VIEW_IND, aggregate: true }
const NO_FILTER: Filters = { task: null, datasets: [], models: [], hparams: {} }

describe('isLowerBetter', () => {
  it('returns true for mse', () => expect(isLowerBetter('mse')).toBe(true))
  it('returns true for cross_entropy_loss', () => expect(isLowerBetter('cross_entropy_loss')).toBe(true))
  it('returns false for accuracy', () => expect(isLowerBetter('accuracy')).toBe(false))
})

describe('useFiltered – individual mode', () => {
  it('returns all rows when no filters active', () => {
    const entries = [makeEntry({ id: 'a' }), makeEntry({ id: 'b' })]
    const { result } = renderHook(() => useFiltered(entries, NO_FILTER, VIEW_IND))
    expect(result.current).toHaveLength(2)
    expect(result.current[0].isAggregated).toBe(false)
  })

  it('filters by task', () => {
    const entries = [
      makeEntry({ id: 'a', task: 'Forecast' }),
      makeEntry({ id: 'b', task: 'UEAClassification' }),
    ]
    const filters: Filters = { ...NO_FILTER, task: 'Forecast' }
    const { result } = renderHook(() => useFiltered(entries, filters, VIEW_IND))
    expect(result.current).toHaveLength(1)
    expect(result.current[0].task).toBe('Forecast')
  })

  it('filters by dataset (multi-select)', () => {
    const entries = [
      makeEntry({ id: 'a', dataset: 'ETTh1' }),
      makeEntry({ id: 'b', dataset: 'ETTm1' }),
    ]
    const filters: Filters = { ...NO_FILTER, datasets: ['ETTh1'] }
    const { result } = renderHook(() => useFiltered(entries, filters, VIEW_IND))
    expect(result.current).toHaveLength(1)
  })

  it('filters by hparam value', () => {
    const entries = [
      makeEntry({ id: 'a', hparams: { windows: 96, pred_len: 96, horizon: 1 } }),
      makeEntry({ id: 'b', hparams: { windows: 336, pred_len: 96, horizon: 1 } }),
    ]
    const filters: Filters = { ...NO_FILTER, hparams: { windows: '96' } }
    const { result } = renderHook(() => useFiltered(entries, filters, VIEW_IND))
    expect(result.current).toHaveLength(1)
    expect(result.current[0].hparams.windows).toBe(96)
  })
})

describe('useFiltered – aggregate mode', () => {
  it('groups two seeds into one row', () => {
    const entries = [
      makeEntry({ id: 'a', seed: 1, metrics: { mse: 0.40 } }),
      makeEntry({ id: 'b', seed: 2, metrics: { mse: 0.36 } }),
    ]
    const { result } = renderHook(() => useFiltered(entries, NO_FILTER, VIEW_AGG))
    expect(result.current).toHaveLength(1)
    expect(result.current[0].isAggregated).toBe(true)
    expect(result.current[0].num_seeds).toBe(2)
    const mse = result.current[0].metrics['mse'] as { mean: number; std: number }
    expect(mse.mean).toBeCloseTo(0.38, 5)
  })

  it('keeps different hparam configs as separate rows', () => {
    const entries = [
      makeEntry({ id: 'a', hparams: { windows: 96, pred_len: 96, horizon: 1 } }),
      makeEntry({ id: 'b', hparams: { windows: 336, pred_len: 96, horizon: 1 } }),
    ]
    const { result } = renderHook(() => useFiltered(entries, NO_FILTER, VIEW_AGG))
    expect(result.current).toHaveLength(2)
  })
})

describe('getBestWorst', () => {
  it('returns min value as best for lower-is-better metrics', () => {
    const rows = [
      { ...makeEntry({ metrics: { mse: 0.4 } }), key: 'a', isAggregated: false, num_seeds: 1 },
      { ...makeEntry({ metrics: { mse: 0.3 } }), key: 'b', isAggregated: false, num_seeds: 1 },
    ]
    const { best, worst } = getBestWorst(rows as never, 'mse')
    expect(best).toBe(0.3)
    expect(worst).toBe(0.4)
  })
})
