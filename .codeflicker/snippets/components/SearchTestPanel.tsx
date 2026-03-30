/**
 * 可复用组件 - SearchTestPanel
 * 
 * @category components
 * @tags components, react, typescript, dependencies
 * @dependencies antd, @ant-design/icons
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/documents/SearchTestPanel.tsx
 * 评分: 4.40
 * 复杂度: 2
 */

export const SearchTestPanel: React.FC = () => {
  const { search, searchResults, loading } = useDocumentStore();
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);

  const handleSearch = async () => {
    if (!query.trim()) return;
    await search(query.trim(), topK);
  };

  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="输入检索查询..."
            onPressEnter={handleSearch}
            style={{ flex: 1 }}
          />
          <Button
            type="primary"
            icon={<SearchOutlined />}
            onClick={handleSearch}
            loading={loading}
          >
            检索
          </Button>
        </div>
        <Space>
          <span style={{ fontSize: 12, color: '#666' }}>返回结果数：{topK}</span>
          <Slider
            min={1}
            max={20}
            value={topK}
            onChange={setTopK}
            style={{ width: 120 }}
          />
        </Space>
      </div>

      {searchResults.length > 0 && (
        <List
          size="small"
          header={<span style={{ fontWeight: 500 }}>检索结果（{searchResults.length}）</span>}
          dataSource={searchResults}
          renderItem={(item, i) => (
            <List.Item>
              <List.Item.Meta
                title={
                  <div style={{ d
// ... 更多实现