import { describe, it, expect } from 'vitest';

describe('Smoke Test', () => {
  it('vitest works', () => {
    expect(1 + 1).toBe(2);
  });

  it('jsdom provides window', () => {
    expect(typeof window).toBe('object');
    expect(typeof document).toBe('object');
  });

  it('Float32Array is available', () => {
    const arr = new Float32Array([1, 2, 3]);
    expect(arr.length).toBe(3);
    expect(arr[0]).toBe(1);
  });
});
