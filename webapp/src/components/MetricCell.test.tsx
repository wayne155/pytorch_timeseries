import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MetricCell } from './MetricCell'

describe('MetricCell', () => {
  it('renders a plain number value', () => {
    render(<MetricCell value={0.3845} isBest={false} isWorst={false} showStd={false} />)
    expect(screen.getByText('0.3845')).toBeInTheDocument()
  })

  it('renders mean for aggregated metric', () => {
    render(<MetricCell value={{ mean: 0.38, std: 0.01 }} isBest={false} isWorst={false} showStd={false} />)
    expect(screen.getByText('0.3800')).toBeInTheDocument()
  })

  it('shows std when showStd is true and value is aggregated', () => {
    render(<MetricCell value={{ mean: 0.38, std: 0.01 }} isBest={false} isWorst={false} showStd={true} />)
    expect(screen.getByText(/±0.0100/)).toBeInTheDocument()
  })

  it('does not show std when showStd is false', () => {
    render(<MetricCell value={{ mean: 0.38, std: 0.01 }} isBest={false} isWorst={false} showStd={false} />)
    expect(screen.queryByText(/±/)).toBeNull()
  })

  it('applies green class when isBest', () => {
    const { container } = render(
      <MetricCell value={0.3} isBest={true} isWorst={false} showStd={false} />
    )
    expect(container.firstChild).toHaveClass('bg-green-50')
  })

  it('applies red class when isWorst', () => {
    const { container } = render(
      <MetricCell value={0.9} isBest={false} isWorst={true} showStd={false} />
    )
    expect(container.firstChild).toHaveClass('bg-red-50')
  })
})
