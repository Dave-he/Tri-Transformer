import { describe, it, expect } from 'vitest';

describe('Project Setup', () => {
  it('should be able to run tests', () => {
    expect(true).toBe(true);
  });

  it('should support TypeScript', () => {
    const value: string = 'hello';
    expect(typeof value).toBe('string');
  });

  it('should support modern JS features', () => {
    const arr = [1, 2, 3];
    const doubled = arr.map((x) => x * 2);
    expect(doubled).toEqual([2, 4, 6]);
  });
});
