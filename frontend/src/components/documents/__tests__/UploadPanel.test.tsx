import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

vi.mock('../../../store/documentStore', () => ({
  useDocumentStore: vi.fn(() => ({
    upload: vi.fn(),
    uploadProgress: null,
    documents: [],
  })),
}));

describe('UploadPanel', () => {
  it('renders upload area', async () => {
    const { UploadPanel } = await import('../UploadPanel');
    render(<UploadPanel />);
    expect(screen.getByText(/上传/i)).toBeDefined();
  });

  it('shows progress bar when uploadProgress is set', async () => {
    const { useDocumentStore } = await import('../../../store/documentStore');
    vi.mocked(useDocumentStore).mockReturnValue({
      upload: vi.fn(),
      uploadProgress: 50,
      documents: [],
      deleteDocument: vi.fn(),
      search: vi.fn(),
      searchResults: [],
    } as never);

    const { UploadPanel } = await import('../UploadPanel');
    render(<UploadPanel />);

    expect(screen.getByRole('progressbar')).toBeDefined();
  });

  it('shows error for unsupported file format', async () => {
    const { useDocumentStore } = await import('../../../store/documentStore');
    const mockUpload = vi.fn().mockRejectedValue(new Error('Unsupported format'));
    vi.mocked(useDocumentStore).mockReturnValue({
      upload: mockUpload,
      uploadProgress: null,
      documents: [],
      deleteDocument: vi.fn(),
      search: vi.fn(),
      searchResults: [],
    } as never);

    const { UploadPanel } = await import('../UploadPanel');
    render(<UploadPanel />);

    const input = screen.getByTestId('file-input');
    const file = new File(['content'], 'test.exe', { type: 'application/octet-stream' });
    Object.defineProperty(input, 'files', { value: [file] });
    fireEvent.change(input);

    expect(screen.getByText('不支持的文件格式，仅支持：.pdf, .docx, .txt, .md')).toBeDefined();
  });
});
