import React from 'react';
import { Form, InputNumber, Select, Button, Space } from 'antd';
import type { TrainingConfig } from '@/types/trainingConfig';

interface TrainingConfigFormProps {
  onSubmit: (config: TrainingConfig) => void;
  loading: boolean;
  initialValues?: Partial<TrainingConfig>;
}

const phaseOptions = [
  { value: 0, label: '阶段 0：缝合层热启动' },
  { value: 1, label: '阶段 1：模态与特征对齐' },
  { value: 2, label: '阶段 2：控制与 RAG 约束训练' },
];

export const TrainingConfigForm: React.FC<TrainingConfigFormProps> = ({
  onSubmit,
  loading,
  initialValues,
}) => {
  const [form] = Form.useForm<TrainingConfig>();

  const handleFinish = (values: TrainingConfig) => {
    onSubmit(values);
  };

  return (
    <Form
      form={form}
      layout="vertical"
      initialValues={{
        learning_rate: initialValues?.learning_rate ?? 1e-4,
        batch_size: initialValues?.batch_size ?? 8,
        max_steps: initialValues?.max_steps ?? 1000,
        phase: initialValues?.phase ?? 0,
        i_model_id: initialValues?.i_model_id ?? '',
        o_model_id: initialValues?.o_model_id ?? '',
      }}
      onFinish={handleFinish}
    >
      <Form.Item
        label="学习率"
        name="learning_rate"
        rules={[
          { required: true, message: '学习率不能为空' },
          {
            validator: (_, value: number) => {
              if (value < 1e-6 || value > 0.1) {
                return Promise.reject(new Error('学习率需在 1e-6 ~ 0.1 之间'));
              }
              return Promise.resolve();
            },
          },
        ]}
      >
        <InputNumber min={1e-6} max={0.1} step={1e-5} style={{ width: '100%' }} />
      </Form.Item>

      <Form.Item
        label="批次大小"
        name="batch_size"
        rules={[{ required: true, message: '批次大小不能为空' }]}
      >
        <InputNumber min={1} max={256} step={1} style={{ width: '100%' }} />
      </Form.Item>

      <Form.Item
        label="最大步数"
        name="max_steps"
        rules={[{ required: true, message: '最大步数不能为空' }]}
      >
        <InputNumber min={1} max={100000} step={100} style={{ width: '100%' }} />
      </Form.Item>

      <Form.Item label="训练阶段" name="phase">
        <Select options={phaseOptions} />
      </Form.Item>

      <Form.Item>
        <Space>
          <Button type="primary" htmlType="submit" loading={loading} disabled={loading}>
            启动训练
          </Button>
        </Space>
      </Form.Item>
    </Form>
  );
};
