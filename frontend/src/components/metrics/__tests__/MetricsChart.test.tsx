import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MetricsChart } from '../MetricsChart';
import type { MetricsHistory } from '@/types/api';

const mockHistory: MetricsHistory[] = [
  { timestamp: '2024-01-01T00:00:00Z', retrievalAccuracy: 0.85, bleuScore: 0.70, hallucinationRate: 0.08 },
  { timestamp: '2024-01-02T00:00:00Z', retrievalAccuracy: 0.87, bleuScore: 0.72, hallucinationRate: 0.07 },
  { timestamp: '2024-01-03T00:00:00Z', retrievalAccuracy: 0.90, bleuScore: 0.75, hallucinationRate: 0.05 },
];

describe('MetricsChart', () => {
  it('renders without crashing with valid history data', () => {
    const { container } = render(<MetricsChart history={mockHistory} />);
    expect(container.firstChild).toBeDefined();
  });

  it('renders legend labels for all three metrics', () => {
    render(<MetricsChart history={mockHistory} />);
    expect(screen.getByText('检索准确率')).toBeDefined();
    expect(screen.getByText('BLEU 分数')).toBeDefined();
    expect(screen.getByText('幻觉率')).toBeDefined();
  });

  it('renders without crashing with empty history', () => {
    const { container } = render(<MetricsChart history={[]} />);
    expect(container.firstChild).toBeDefined();
  });
});
