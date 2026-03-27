import React, { useState } from 'react';
import { Input, Button, List, Tag, Space, Slider } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { useDocumentStore } from '@/store/documentStore';

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
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>#{i + 1} {item.document}</span>
                    <Tag color="blue">相关度 {(item.score * 100).toFixed(0)}%</Tag>
                  </div>
                }
                description={<span style={{ fontSize: 12, color: '#666' }}>{item.chunk}</span>}
              />
            </List.Item>
          )}
        />
      )}
    </div>
  );
};
