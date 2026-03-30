/**
 * 可复用组件 - UploadPanel
 * 
 * @category components
 * @tags components, react, typescript, dependencies
 * @dependencies antd, @ant-design/icons
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/documents/UploadPanel.tsx
 * 评分: 4.78
 * 复杂度: 7
 */

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
    
// ... 更多实现