import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { MetricsHistory } from '@/types/api';
import { formatDate } from '@/utils/formatDate';

interface MetricsChartProps {
  history: MetricsHistory[];
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ history }) => {
  const data = history.map((h) => ({
    ...h,
    date: formatDate(h.timestamp),
    retrievalAccuracyPct: (h.retrievalAccuracy * 100).toFixed(1),
    bleuScorePct: (h.bleuScore * 100).toFixed(1),
    hallucinationRatePct: (h.hallucinationRate * 100).toFixed(1),
  }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} unit="%" />
        <Tooltip formatter={(v: unknown) => `${String(v)}%`} />
        <Legend />
        <Line
          type="monotone"
          dataKey="retrievalAccuracyPct"
          name="检索准确率"
          stroke="#1677ff"
          strokeWidth={2}
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="bleuScorePct"
          name="BLEU 分数"
          stroke="#52c41a"
          strokeWidth={2}
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="hallucinationRatePct"
          name="幻觉率"
          stroke="#ff4d4f"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};
