import React, { useEffect } from 'react';
import { Card, Divider } from 'antd';
import { TrainingStatusCard } from '@/components/metrics/TrainingStatusCard';
import { ModelPluginSelector } from '@/components/training/ModelPluginSelector';
import { TrainingConfigForm } from '@/components/training/TrainingConfigForm';
import { useMetricsStore } from '@/store/metricsStore';
import { useTrainingConfigStore } from '@/store/trainingConfigStore';
import type { TrainingConfig } from '@/types/trainingConfig';

const TrainingPage: React.FC = () => {
  const { trainingStatus, loading: metricsLoading, fetchStatus } = useMetricsStore();
  const {
    config,
    availableModels,
    loading,
    startTraining,
    fetchProgress,
    fetchAvailableModels,
    setConfig,
  } = useTrainingConfigStore();

  useEffect(() => {
    fetchStatus().catch(() => undefined);
    fetchAvailableModels().catch(() => undefined);
    const timer = setInterval(() => {
      fetchStatus().catch(() => undefined);
      fetchProgress().catch(() => undefined);
    }, 5000);
    return () => clearInterval(timer);
  }, [fetchStatus, fetchAvailableModels, fetchProgress]);

  const handleSubmit = async (values: TrainingConfig) => {
    setConfig(values);
    await startTraining();
  };

  return (
    <div style={{ padding: 24 }}>
      <TrainingStatusCard status={trainingStatus} loading={metricsLoading} />
      <Divider />
      <Card title="模型插件配置" style={{ marginBottom: 16 }}>
        <ModelPluginSelector
          availableModels={availableModels}
          iModelId={config.i_model_id}
          oModelId={config.o_model_id}
          onIModelChange={(value) => setConfig({ i_model_id: value })}
          onOModelChange={(value) => setConfig({ o_model_id: value })}
        />
      </Card>
      <Card title="训练超参配置">
        <TrainingConfigForm
          onSubmit={handleSubmit}
          loading={loading}
          initialValues={config}
        />
      </Card>
    </div>
  );
};

export default TrainingPage;
