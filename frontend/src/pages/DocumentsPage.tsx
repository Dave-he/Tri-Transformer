import React, { useEffect } from 'react';
import { Card, Tabs } from 'antd';
import { UploadPanel } from '@/components/documents/UploadPanel';
import { DocumentList } from '@/components/documents/DocumentList';
import { SearchTestPanel } from '@/components/documents/SearchTestPanel';
import { useDocumentStore } from '@/store/documentStore';

const DocumentsPage: React.FC = () => {
  const { documents, loading, fetchDocuments, deleteDocument } = useDocumentStore();

  useEffect(() => {
    fetchDocuments().catch(() => undefined);
  }, [fetchDocuments]);

  return (
    <div>
      <Card title="知识库管理" style={{ marginBottom: 16 }}>
        <UploadPanel />
      </Card>
      <Tabs
        items={[
          {
            key: 'documents',
            label: `文档列表（${documents.length}）`,
            children: (
              <DocumentList documents={documents} onDelete={deleteDocument} loading={loading} />
            ),
          },
          {
            key: 'search',
            label: '检索测试',
            children: <SearchTestPanel />,
          },
        ]}
      />
    </div>
  );
};

export default DocumentsPage;
