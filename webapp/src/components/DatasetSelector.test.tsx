import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { DatasetSelector } from './DatasetSelector'

const DATASETS = ['ETTh1', 'ETTh2', 'ETTm1']

describe('DatasetSelector', () => {
  it('renders All option and all datasets', () => {
    render(
      <DatasetSelector datasets={DATASETS} selected="All" onChange={() => {}} />
    )
    expect(screen.getByLabelText('All')).toBeInTheDocument()
    DATASETS.forEach(d => expect(screen.getByLabelText(d)).toBeInTheDocument())
  })

  it('marks the selected dataset as checked', () => {
    render(
      <DatasetSelector datasets={DATASETS} selected="ETTh2" onChange={() => {}} />
    )
    expect(screen.getByLabelText('ETTh2')).toBeChecked()
    expect(screen.getByLabelText('All')).not.toBeChecked()
  })

  it('calls onChange when a dataset is clicked', () => {
    const onChange = vi.fn()
    render(
      <DatasetSelector datasets={DATASETS} selected="All" onChange={onChange} />
    )
    fireEvent.click(screen.getByLabelText('ETTh1'))
    expect(onChange).toHaveBeenCalledWith('ETTh1')
  })

  it('calls onChange with All when All is clicked', () => {
    const onChange = vi.fn()
    render(
      <DatasetSelector datasets={DATASETS} selected="ETTh1" onChange={onChange} />
    )
    fireEvent.click(screen.getByLabelText('All'))
    expect(onChange).toHaveBeenCalledWith('All')
  })
})
