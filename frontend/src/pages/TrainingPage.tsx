import React, { useEffect } from 'react';
import { TrainingStatusCard } from '@/components/metrics/TrainingStatusCard';
import { useMetricsStore } from '@/store/metricsStore';

const TrainingPage: React.FC = () => {
  const { trainingStatus, loading, fetchStatus } = useMetricsStore();

  useEffect(() => {
    fetchStatus().catch(() => undefined);
    const timer = setInterval(() => {
      fetchStatus().catch(() => undefined);
    }, 10000);
    return () => clearInterval(timer);
  }, [fetchStatus]);

  return (
    <div>
      <TrainingStatusCard status={trainingStatus} loading={loading} />
    </div>
  );
};

export default TrainingPage;
