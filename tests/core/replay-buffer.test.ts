import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  ReplayBuffer,
  PrioritizedReplayBuffer
} from '../../src/core/replay-buffer.js';

describe('ReplayBuffer', () => {
  let buffer: ReplayBuffer;
  const state = (i: number): Float32Array => new Float32Array([i, i + 0.5, i + 1.0]);
  const nextState = (i: number): Float32Array => new Float32Array([i + 0.1, i + 0.6, i + 1.1]);

  describe('constructor', () => {
    it('initializes with default capacity 10000', () => {
      buffer = new ReplayBuffer();
      expect(buffer.capacity).toBe(10_000);
      expect(buffer.length()).toBe(0);
    });

    it('accepts custom capacity', () => {
      buffer = new ReplayBuffer(100);
      expect(buffer.capacity).toBe(100);
    });

    it('throws on non-positive capacity', () => {
      expect(() => new ReplayBuffer(0)).toThrow();
      expect(() => new ReplayBuffer(-1)).toThrow();
    });
  });

  describe('add()', () => {
    beforeEach(() => {
      buffer = new ReplayBuffer(5);
    });

    it('adds a single experience', () => {
      buffer.add(state(1), 0, 1.0, nextState(1), false);
      expect(buffer.length()).toBe(1);
    });

    it('copies Float32Array (no shared reference)', () => {
      const s = state(1);
      const ns = nextState(1);
      buffer.add(s, 0, 1.0, ns, false);
      // Mutate originals — stored copies must be unaffected
      s[0] = 999;
      ns[0] = 999;
      const recent = buffer.getRecent(1);
      expect(recent[0]!.state[0]).toBe(1);
      expect(recent[0]!.nextState[0]).toBeCloseTo(1.1, 5);
    });

    it('overwrites oldest experience when capacity reached (circular)', () => {
      for (let i = 0; i < 7; i++) {
        buffer.add(state(i), i, i * 10, nextState(i), false);
      }
      expect(buffer.length()).toBe(5);
    });

    it('preserves all data fields', () => {
      buffer.add(state(1), 7, 42.5, nextState(1), true);
      const recent = buffer.getRecent(1);
      expect(recent[0]!.action).toBe(7);
      expect(recent[0]!.reward).toBe(42.5);
      expect(recent[0]!.done).toBe(true);
    });
  });

  describe('sample()', () => {
    beforeEach(() => {
      buffer = new ReplayBuffer(10);
      for (let i = 0; i < 5; i++) {
        buffer.add(state(i), i, i, nextState(i), false);
      }
    });

    it('returns batch of requested size', () => {
      const batch = buffer.sample(3);
      expect(batch.states).toHaveLength(3);
      expect(batch.actions).toHaveLength(3);
      expect(batch.rewards).toHaveLength(3);
      expect(batch.nextStates).toHaveLength(3);
      expect(batch.dones).toHaveLength(3);
    });

    it('throws when batchSize > size', () => {
      expect(() => buffer.sample(10)).toThrow(/Cannot sample 10/);
    });

    it('throws when buffer empty', () => {
      const empty = new ReplayBuffer(10);
      expect(() => empty.sample(1)).toThrow(/only 0 available/);
    });

    it('returns dones as numbers (0/1), not booleans', () => {
      buffer.add(state(99), 0, 0, nextState(99), true);
      const batch = buffer.sample(buffer.length());
      batch.dones.forEach((d) => {
        expect([0, 1]).toContain(d);
      });
      expect(batch.dones).toContain(1);
    });

    it('samples without replacement (mocked random always returns 0)', () => {
      // Mock Math.random to always return 0 — without replacement, picks must differ
      const spy = vi.spyOn(Math, 'random').mockReturnValue(0);
      try {
        const batch = buffer.sample(3);
        const uniqueActions = new Set(batch.actions);
        expect(uniqueActions.size).toBe(3);
      } finally {
        spy.mockRestore();
      }
    });
  });

  describe('canSample()', () => {
    it('returns false when size < batchSize', () => {
      buffer = new ReplayBuffer(10);
      buffer.add(state(0), 0, 0, nextState(0), false);
      expect(buffer.canSample(5)).toBe(false);
    });

    it('returns true when size >= batchSize', () => {
      buffer = new ReplayBuffer(10);
      for (let i = 0; i < 5; i++) {
        buffer.add(state(i), i, i, nextState(i), false);
      }
      expect(buffer.canSample(5)).toBe(true);
      expect(buffer.canSample(3)).toBe(true);
    });
  });

  describe('clear()', () => {
    it('resets all state', () => {
      buffer = new ReplayBuffer(5);
      for (let i = 0; i < 3; i++) {
        buffer.add(state(i), i, i, nextState(i), false);
      }
      buffer.clear();
      expect(buffer.length()).toBe(0);
    });
  });

  describe('getRecent()', () => {
    beforeEach(() => {
      buffer = new ReplayBuffer(10);
    });

    it('returns last N experiences (default 10)', () => {
      for (let i = 0; i < 5; i++) {
        buffer.add(state(i), i, i, nextState(i), false);
      }
      const recent = buffer.getRecent();
      expect(recent).toHaveLength(5);
    });

    it('returns only n most recent', () => {
      for (let i = 0; i < 5; i++) {
        buffer.add(state(i), i, i, nextState(i), false);
      }
      const recent = buffer.getRecent(2);
      expect(recent).toHaveLength(2);
    });

    it('handles n > size (returns all)', () => {
      buffer.add(state(0), 0, 0, nextState(0), false);
      buffer.add(state(1), 1, 1, nextState(1), false);
      const recent = buffer.getRecent(10);
      expect(recent).toHaveLength(2);
    });
  });

  describe('getStats()', () => {
    it('returns zeros for empty buffer', () => {
      buffer = new ReplayBuffer(10);
      const stats = buffer.getStats();
      expect(stats).toEqual({
        size: 0,
        capacity: 10,
        utilizationPercent: 0,
        avgReward: 0,
        maxReward: 0,
        minReward: 0
      });
    });

    it('computes reward statistics', () => {
      buffer = new ReplayBuffer(10);
      buffer.add(state(0), 0, 10, nextState(0), false);
      buffer.add(state(1), 1, 20, nextState(1), false);
      buffer.add(state(2), 2, 30, nextState(2), false);
      const stats = buffer.getStats();
      expect(stats.size).toBe(3);
      expect(stats.avgReward).toBe(20);
      expect(stats.maxReward).toBe(30);
      expect(stats.minReward).toBe(10);
      expect(stats.utilizationPercent).toBe(30);
    });
  });

  describe('toJSON() / fromJSON()', () => {
    it('round-trips a non-empty buffer', () => {
      buffer = new ReplayBuffer(5);
      for (let i = 0; i < 3; i++) {
        buffer.add(state(i), i, i * 5, nextState(i), false);
      }
      const json = buffer.toJSON();
      const restored = new ReplayBuffer();
      restored.fromJSON(json);
      expect(restored.length()).toBe(buffer.length());
      expect(restored.capacity).toBe(buffer.capacity);
      const origBatch = buffer.sample(3);
      const restBatch = restored.sample(3);
      expect(origBatch.actions.sort()).toEqual(restBatch.actions.sort());
    });

    it('serializes Float32Arrays as plain arrays', () => {
      buffer = new ReplayBuffer(5);
      buffer.add(state(1), 0, 1, nextState(1), false);
      const json = buffer.toJSON();
      expect(Array.isArray(json.buffer[0]!.state)).toBe(true);
      expect(json.buffer[0]!.state).toEqual([1, 1.5, 2.0]);
    });

    it('fromJSON restores Float32Array type', () => {
      buffer = new ReplayBuffer(5);
      buffer.add(state(1), 0, 1, nextState(1), false);
      const restored = new ReplayBuffer();
      restored.fromJSON(buffer.toJSON());
      const recent = restored.getRecent(1);
      expect(recent[0]!.state).toBeInstanceOf(Float32Array);
      expect(recent[0]!.nextState).toBeInstanceOf(Float32Array);
    });
  });

  describe('generic state type (non-Float32Array)', () => {
    it('accepts custom state type with explicit clone', () => {
      // Custom state type: simple object
      type CustomState = { x: number; y: number };
      const clone = (s: CustomState): CustomState => ({ x: s.x, y: s.y });
      const customBuffer = new ReplayBuffer<CustomState>(5, { cloneState: clone });
      customBuffer.add({ x: 1, y: 2 }, 0, 1, { x: 3, y: 4 }, false);
      expect(customBuffer.length()).toBe(1);
      const recent = customBuffer.getRecent(1);
      expect(recent[0]!.state).toEqual({ x: 1, y: 2 });
    });
  });
});

describe('PrioritizedReplayBuffer', () => {
  let buffer: PrioritizedReplayBuffer;
  const state = (i: number): Float32Array => new Float32Array([i, i + 0.5]);
  const nextState = (i: number): Float32Array => new Float32Array([i + 0.1, i + 0.6]);

  beforeEach(() => {
    buffer = new PrioritizedReplayBuffer(10, { alpha: 0.6, beta: 0.4 });
  });

  it('inherits from ReplayBuffer', () => {
    expect(buffer).toBeInstanceOf(ReplayBuffer);
    expect(buffer.capacity).toBe(10);
  });

  it('stores alpha and beta parameters', () => {
    expect(buffer.alpha).toBe(0.6);
    expect(buffer.beta).toBe(0.4);
  });

  it('add() sets max priority for new experience', () => {
    (buffer as unknown as { maxPriority: number }).maxPriority = 5.0;
    buffer.add(state(0), 0, 1, nextState(0), false);
    const lastIdx = (buffer['position'] - 1 + buffer.capacity) % buffer.capacity;
    // Accessing protected member for test verification
    expect((buffer as unknown as { priorities: Float32Array }).priorities[lastIdx]).toBe(5.0);
  });

  describe('sampleWithPriorities()', () => {
    beforeEach(() => {
      for (let i = 0; i < 8; i++) {
        buffer.add(state(i), i, i, nextState(i), false);
      }
    });

    it('returns batch with correct shape', () => {
      const result = buffer.sampleWithPriorities(3);
      expect(result.batch.states).toHaveLength(3);
      expect(result.indices).toHaveLength(3);
      expect(result.weights).toHaveLength(3);
    });

    it('returns weights in [0, 1] range (normalized)', () => {
      const { weights } = buffer.sampleWithPriorities(4);
      for (const w of weights) {
        expect(w).toBeGreaterThanOrEqual(0);
        expect(w).toBeLessThanOrEqual(1);
      }
      // At least one weight is 1.0 (the max-normalized one)
      expect(Math.max(...weights)).toBeCloseTo(1.0, 5);
    });

    it('samples without replacement', () => {
      const { indices } = buffer.sampleWithPriorities(3);
      expect(new Set(indices).size).toBe(3);
    });

    it('throws when batchSize > size', () => {
      expect(() => buffer.sampleWithPriorities(20)).toThrow();
    });
  });

  describe('updatePriorities()', () => {
    it('updates priorities and tracks max', () => {
      buffer.updatePriorities([0, 1, 2], [0.5, 2.0, 1.0]);
      const internal = buffer as unknown as { priorities: Float32Array; maxPriority: number };
      expect(internal.priorities[0]).toBeCloseTo(0.5 + 1e-6, 7);
      expect(internal.priorities[1]).toBeCloseTo(2.0 + 1e-6, 7);
      expect(internal.priorities[2]).toBeCloseTo(1.0 + 1e-6, 7);
      expect(internal.maxPriority).toBeCloseTo(2.0 + 1e-6, 7);
    });

    it('uses abs() of priority + epsilon', () => {
      buffer.updatePriorities([0], [-3.5]);
      const internal = buffer as unknown as { priorities: Float32Array };
      expect(internal.priorities[0]).toBeCloseTo(3.5 + 1e-6, 7);
    });
  });
});
