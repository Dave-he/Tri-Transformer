import React from 'react';
import { Table, Tag, Button, Popconfirm } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import { formatDate } from '@/utils/formatDate';
import type { Document, DocumentStatus } from '@/types/api';

const statusConfig: Record<DocumentStatus, { color: string; label: string }> = {
  processing: { color: 'processing', label: '处理中' },
  ready: { color: 'success', label: '可用' },
  failed: { color: 'error', label: '失败' },
};

interface DocumentListProps {
  documents: Document[];
  onDelete: (id: string) => void;
  loading?: boolean;
}

export const DocumentList: React.FC<DocumentListProps> = ({ documents, onDelete, loading }) => {
  const columns = [
    { title: '文件名', dataIndex: 'name', key: 'name', ellipsis: true },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 80,
      render: (t: string) => <Tag>{t.toUpperCase()}</Tag>,
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      width: 100,
      render: (s: number) => `${(s / 1024).toFixed(1)} KB`,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: DocumentStatus) => {
        const cfg = statusConfig[status];
        return <Tag color={cfg.color}>{cfg.label}</Tag>;
      },
    },
    {
      title: '上传时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 120,
      render: (d: string) => formatDate(d),
    },
    {
      title: '操作',
      key: 'action',
      width: 80,
      render: (_: unknown, record: Document) => (
        <Popconfirm title="确认删除此文档？" onConfirm={() => onDelete(record.id)}>
          <Button type="text" danger icon={<DeleteOutlined />} size="small" />
        </Popconfirm>
      ),
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={documents}
      rowKey="id"
      loading={loading}
      pagination={{ pageSize: 10, showTotal: (total) => `共 ${total} 个文档` }}
      size="small"
    />
  );
};
