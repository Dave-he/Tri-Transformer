import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import type { MetricsHistory } from '@/types/api';

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) =>
    React.createElement('div', { 'data-testid': 'chart-container' }, children),
  LineChart: ({ children, 'data-testid': t }: { children: React.ReactNode; 'data-testid'?: string }) =>
    React.createElement('div', { 'data-testid': t ?? 'line-chart' }, children),
  Line: ({ name }: { name: string }) =>
    React.createElement('span', { 'data-line-name': name }),
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

import React from 'react';
import { MetricsChart } from '../MetricsChart';

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

  it('renders three Line series with correct names', () => {
    const { container } = render(<MetricsChart history={mockHistory} />);
    const lines = container.querySelectorAll('[data-line-name]');
    const names = Array.from(lines).map((el) => el.getAttribute('data-line-name'));
    expect(names).toContain('检索准确率');
    expect(names).toContain('BLEU 分数');
    expect(names).toContain('幻觉率');
  });

  it('renders without crashing with empty history', () => {
    const { container } = render(<MetricsChart history={[]} />);
    expect(container.firstChild).toBeDefined();
  });
});
