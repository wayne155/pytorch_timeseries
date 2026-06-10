import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { TaskTable } from './TaskTable'
import type { TaskTableRow, TaskViewOptions } from '../types'
import type { ColumnDef } from '../hooks/useTaskView'

const DEFAULT_OPTS: TaskViewOptions = {
  showStd: false, sortColumn: null, sortMetric: null, sortDirection: 'asc',
}

function makeRow(name: string, avgMse = 0.4): TaskTableRow {
  return {
    model: name,
    source_type: 'local_run',
    citation: '',
    url: '',
    columns: {
      avg: { mse: { mean: avgMse, std: 0.01, n_seeds: 3 }, mae: { mean: 0.30, std: 0.005, n_seeds: 3 } },
      '96': { mse: { mean: avgMse - 0.03, std: 0.008, n_seeds: 3 }, mae: { mean: 0.27, std: 0.004, n_seeds: 3 } },
    },
  }
}

const COL_DEFS: ColumnDef[] = [
  { id: 'avg', label: 'avg' },
  { id: '96', label: '96' },
]

describe('TaskTable', () => {
  it('renders column group headers (avg, 96)', () => {
    render(
      <TaskTable
        rows={[makeRow('DLinear')]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse', 'mae']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    expect(screen.getAllByText('avg').length).toBeGreaterThan(0)
    expect(screen.getAllByText('96').length).toBeGreaterThan(0)
  })

  it('renders metric sub-headers (mse, mae) under each group', () => {
    render(
      <TaskTable
        rows={[makeRow('DLinear')]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse', 'mae']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    // mse and mae appear once per column group (2 groups × 1 each)
    expect(screen.getAllByText(/mse/i).length).toBe(COL_DEFS.length)
    expect(screen.getAllByText(/mae/i).length).toBe(COL_DEFS.length)
  })

  it('renders model name in a row', () => {
    render(
      <TaskTable
        rows={[makeRow('DLinear'), makeRow('PatchTST', 0.35)]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    expect(screen.getByText('DLinear')).toBeInTheDocument()
    expect(screen.getByText('PatchTST')).toBeInTheDocument()
  })

  it('shows empty state when no rows', () => {
    render(
      <TaskTable
        rows={[]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    expect(screen.getByText(/no results/i)).toBeInTheDocument()
  })

  it('renders [⋯] button per row', () => {
    render(
      <TaskTable
        rows={[makeRow('DLinear'), makeRow('PatchTST')]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    expect(screen.getAllByTitle('Row actions')).toHaveLength(2)
  })

  it('calls onSortChange when metric header is clicked', async () => {
    const onSortChange = vi.fn()
    render(
      <TaskTable
        rows={[makeRow('DLinear')]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={onSortChange}
      />
    )
    // Find the mse header under the "avg" group (there may be multiple mse headers)
    const mseHeaders = screen.getAllByText(/mse/i)
    // Click the first one
    fireEvent.click(mseHeaders[0])
    expect(onSortChange).toHaveBeenCalledWith('avg', 'mse', 'asc')
  })
})
