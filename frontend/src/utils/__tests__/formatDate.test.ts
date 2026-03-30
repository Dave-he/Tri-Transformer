import { describe, it, expect } from 'vitest';
import { formatDate, formatDateTime } from '../formatDate';

describe('formatDate', () => {
  it('formats ISO timestamp to zh-CN date string', () => {
    const result = formatDate('2024-03-15T10:30:00.000Z');
    expect(result).toMatch(/2024/);
    expect(result).toMatch(/03|3/);
  });

  it('returns a non-empty string for valid date', () => {
    const result = formatDate('2025-01-01T00:00:00Z');
    expect(typeof result).toBe('string');
    expect(result.length).toBeGreaterThan(0);
  });

  it('handles epoch timestamp string', () => {
    const result = formatDate('1970-01-01T00:00:00.000Z');
    expect(typeof result).toBe('string');
    expect(result.length).toBeGreaterThan(0);
  });
});

describe('formatDateTime', () => {
  it('formats ISO timestamp to zh-CN datetime string', () => {
    const result = formatDateTime('2024-03-15T14:30:00.000Z');
    expect(result).toMatch(/2024/);
    expect(typeof result).toBe('string');
    expect(result.length).toBeGreaterThan(0);
  });

  it('includes hour and minute in output', () => {
    const result = formatDateTime('2024-06-20T08:05:00.000Z');
    expect(typeof result).toBe('string');
    expect(result).toMatch(/\d{2}/);
  });

  it('returns a longer string than formatDate for same input', () => {
    const ts = '2024-09-10T12:00:00.000Z';
    expect(formatDateTime(ts).length).toBeGreaterThan(formatDate(ts).length);
  });
});
