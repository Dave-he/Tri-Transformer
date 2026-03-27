import React, { useEffect } from 'react';
import { Card, Row, Col, Statistic } from 'antd';
import { MetricsChart } from '@/components/metrics/MetricsChart';
import { useMetricsStore } from '@/store/metricsStore';

const MetricsPage: React.FC = () => {
  const { metrics, loading, fetchMetrics } = useMetricsStore();

  useEffect(() => {
    fetchMetrics().catch(() => undefined);
    const timer = setInterval(() => {
      fetchMetrics().catch(() => undefined);
    }, 10000);
    return () => clearInterval(timer);
  }, [fetchMetrics]);

  return (
    <div>
      {metrics && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={8}>
            <Card>
              <Statistic
                title="检索准确率"
                value={(metrics.current.retrievalAccuracy * 100).toFixed(1)}
                suffix="%"
                valueStyle={{ color: '#1677ff' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="BLEU 分数"
                value={(metrics.current.bleuScore * 100).toFixed(1)}
                suffix="%"
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="幻觉率"
                value={(metrics.current.hallucinationRate * 100).toFixed(1)}
                suffix="%"
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
        </Row>
      )}
      <Card title="历史趋势" loading={loading && !metrics}>
        {metrics?.history && <MetricsChart history={metrics.history} />}
      </Card>
    </div>
  );
};

export default MetricsPage;
