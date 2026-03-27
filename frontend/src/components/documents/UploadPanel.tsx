import React, { useState } from 'react';
import { Alert, Upload } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { useDocumentStore } from '@/store/documentStore';

const { Dragger } = Upload;

const ALLOWED_TYPES = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'text/markdown', 'text/x-markdown'];
const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md'];

export const UploadPanel: React.FC = () => {
  const { upload, uploadProgress } = useDocumentStore();
  const [error, setError] = useState<string | null>(null);

  const isValidFile = (file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    return ALLOWED_EXTENSIONS.includes(ext) || ALLOWED_TYPES.includes(file.type);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!isValidFile(file)) {
      setError(`不支持的文件格式，仅支持：${ALLOWED_EXTENSIONS.join(', ')}`);
      return;
    }
    setError(null);
    upload(file).catch((err: Error) => setError(err.message ?? '上传失败'));
  };

  return (
    <div>
      <Dragger
        accept={ALLOWED_EXTENSIONS.join(',')}
        showUploadList={false}
        customRequest={({ file }) => {
          if (file instanceof File) {
            if (!isValidFile(file)) {
              setError(`不支持的文件格式，仅支持：${ALLOWED_EXTENSIONS.join(', ')}`);
              return;
            }
            setError(null);
            upload(file).catch((err: Error) => setError(err.message ?? '上传失败'));
          }
        }}
        style={{ marginBottom: 12 }}
      >
        <p className="ant-upload-drag-icon"><InboxOutlined /></p>
        <p>拖拽文件到此处，或点击选择文件上传</p>
        <p style={{ fontSize: 12, color: '#999' }}>支持 {ALLOWED_EXTENSIONS.join('、')} 格式</p>
      </Dragger>
      <input
        data-testid="file-input"
        type="file"
        accept={ALLOWED_EXTENSIONS.join(',')}
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      {uploadProgress !== null && (
        <div role="progressbar" aria-valuenow={uploadProgress} style={{ marginTop: 8 }}>
          <div style={{ height: 8, background: '#f0f0f0', borderRadius: 4 }}>
            <div style={{ height: '100%', width: `${uploadProgress}%`, background: '#1677ff', borderRadius: 4, transition: 'width 0.3s' }} />
          </div>
        </div>
      )}
      {error && <Alert message={error} type="error" style={{ marginTop: 8 }} closable onClose={() => setError(null)} />}
    </div>
  );
};

