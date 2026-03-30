import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';

vi.mock('../../../store/documentStore', () => ({
  useDocumentStore: vi.fn(() => ({
    upload: vi.fn(),
    uploadProgress: null,
    documents: [],
  })),
}));

import { UploadPanel } from '../UploadPanel';
import { useDocumentStore } from '../../../store/documentStore';

describe('UploadPanel', () => {
  beforeEach(() => {
    cleanup();
    vi.mocked(useDocumentStore).mockReturnValue({
      upload: vi.fn(),
      uploadProgress: null,
      documents: [],
      deleteDocument: vi.fn(),
      search: vi.fn(),
      searchResults: [],
    } as never);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('renders upload area', () => {
    render(<UploadPanel />);
    expect(screen.getByText(/上传/i)).toBeDefined();
  });

  it('shows progress bar when uploadProgress is set', () => {
    vi.mocked(useDocumentStore).mockReturnValue({
      upload: vi.fn(),
      uploadProgress: 50,
      documents: [],
      deleteDocument: vi.fn(),
      search: vi.fn(),
      searchResults: [],
    } as never);

    render(<UploadPanel />);

    expect(screen.getByTestId('upload-progress')).toBeDefined();
  });

  it('shows error for unsupported file format', () => {
    const mockUpload = vi.fn().mockRejectedValue(new Error('Unsupported format'));
    vi.mocked(useDocumentStore).mockReturnValue({
      upload: mockUpload,
      uploadProgress: null,
      documents: [],
      deleteDocument: vi.fn(),
      search: vi.fn(),
      searchResults: [],
    } as never);

    render(<UploadPanel />);

    const input = screen.getByTestId('file-input');
    const file = new File(['content'], 'test.exe', { type: 'application/octet-stream' });
    Object.defineProperty(input, 'files', { value: [file] });
    fireEvent.change(input);

    expect(screen.getByText('不支持的文件格式，仅支持：.pdf, .docx, .txt, .md')).toBeDefined();
  });
});
