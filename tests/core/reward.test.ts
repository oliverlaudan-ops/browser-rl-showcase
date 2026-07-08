import { describe, it, expect } from 'vitest';
import {
  calculateReturn,
  calculateNStepReturn,
  normalizeRewards,
  clipRewards,
  getRewardStats,
  calculateAdvantages,
  potentialShaping,
  linearPotential,
  progressShapingReward,
  type Transition
} from '../../src/core/reward.js';

describe('calculateReturn', () => {
  it('returns 0 for empty rewards', () => {
    expect(calculateReturn([])).toBe(0);
  });

  it('returns first reward only when gamma=0 (degenerate case)', () => {
    // G_t = r_t + 0·G_{t+1} → final G after backward loop = rewards[0]
    expect(calculateReturn([1, 2, 3, 4], 0)).toBe(1);
  });

  it('applies discount backward when gamma=0.5', () => {
    // rewards = [1, 1, 1], gamma = 0.5
    // Backward: G = 1, G = 1+0.5·1=1.5, G = 1+0.5·1.5=1.75
    expect(calculateReturn([1, 1, 1], 0.5)).toBeCloseTo(1.75, 5);
  });

  it('uses gamma=0.99 by default', () => {
    expect(calculateReturn([1, 1, 1])).toBeCloseTo(
      calculateReturn([1, 1, 1], 0.99),
      10
    );
  });
});

describe('calculateNStepReturn', () => {
  it('caps at n steps (no future bleed)', () => {
    // n=2 of [1, 1, 1, 1] with gamma=1: G = 1 + 1 = 2
    expect(calculateNStepReturn([1, 1, 1, 1], 2, 1)).toBe(2);
  });

  it('handles n > rewards.length', () => {
    // n=10 of [1, 1] with gamma=0: G = 1 (first only)
    expect(calculateNStepReturn([1, 1], 10, 0)).toBe(1);
  });

  it('handles n=0 (returns 0)', () => {
    expect(calculateNStepReturn([1, 2, 3], 0)).toBe(0);
  });
});

describe('normalizeRewards', () => {
  it('returns empty array for empty input', () => {
    expect(normalizeRewards([])).toEqual([]);
  });

  it('centers rewards around mean 0', () => {
    const normalized = normalizeRewards([1, 2, 3, 4, 5]);
    const mean = normalized.reduce((s, v) => s + v, 0) / normalized.length;
    expect(mean).toBeCloseTo(0, 5);
    // Reference to silence noUnusedLocals
    void mean;
  });

  it('produces unit variance', () => {
    const normalized = normalizeRewards([1, 2, 3, 4, 5]);
    const variance = normalized.reduce((s, v) => s + v ** 2, 0) / normalized.length;
    expect(Math.sqrt(variance)).toBeCloseTo(1, 1);
  });

  it('handles uniform rewards (no NaN)', () => {
    const normalized = normalizeRewards([5, 5, 5]);
    normalized.forEach((v) => {
      expect(isFinite(v)).toBe(true);
      expect(isNaN(v)).toBe(false);
    });
  });
});

describe('clipRewards', () => {
  it('clamps values above maxAbs', () => {
    expect(clipRewards([10, -10, 1, -1], 5)).toEqual([5, -5, 1, -1]);
  });

  it('passes through values within range unchanged', () => {
    expect(clipRewards([1, 2, 3], 5)).toEqual([1, 2, 3]);
  });

  it('handles empty array', () => {
    expect(clipRewards([], 1)).toEqual([]);
  });
});

describe('getRewardStats', () => {
  it('returns zeros for empty array', () => {
    expect(getRewardStats([])).toEqual({
      total: 0,
      mean: 0,
      min: 0,
      max: 0,
      std: 0,
      count: 0
    });
  });

  it('computes total/mean/min/max/std for known data', () => {
    const stats = getRewardStats([1, 2, 3, 4, 5]);
    expect(stats.total).toBe(15);
    expect(stats.mean).toBe(3);
    expect(stats.min).toBe(1);
    expect(stats.max).toBe(5);
    expect(stats.count).toBe(5);
    expect(stats.std).toBeCloseTo(Math.sqrt(2), 5);
  });

  it('handles single-element array', () => {
    const stats = getRewardStats([42]);
    expect(stats.total).toBe(42);
    expect(stats.mean).toBe(42);
    expect(stats.std).toBe(0);
    expect(stats.count).toBe(1);
  });
});

describe('calculateAdvantages (GAE)', () => {
  it('returns empty array for empty rewards', () => {
    expect(calculateAdvantages([], [])).toEqual([]);
  });

  it('matches manual GAE computation', () => {
    // 3 timesteps, values = V(s_0..s_3), rewards at t=0..2
    const rewards = [1, 1, 1];
    const values = [0.5, 0.6, 0.7, 0.0];
    const adv = calculateAdvantages(rewards, values, 0.99, 0.95);
    // t=2: δ = 1 + 0.99·0 - 0.7 = 0.3, A=0.3
    // t=1: δ = 1 + 0.99·0.7 - 0.6 = 1.093, A = 1.093 + 0.99·0.95·0.3 = 1.375
    // t=0: δ = 1 + 0.99·0.6 - 0.5 = 1.094, A = 1.094 + 0.99·0.95·1.375 = 2.388
    expect(adv).toHaveLength(3);
    expect(adv[2]).toBeCloseTo(0.3, 3);
    expect(adv[1]).toBeCloseTo(1.375, 2);
    expect(adv[0]).toBeCloseTo(2.388, 2);
  });

  it('default lambda=0.95 and gamma=0.99', () => {
    const adv1 = calculateAdvantages([1], [0, 0]);
    const adv2 = calculateAdvantages([1], [0, 0], 0.99, 0.95);
    expect(adv1).toEqual(adv2);
  });
});

describe('potentialShaping', () => {
  it('F = γ·Φ(s\') - Φ(s) added to base reward', () => {
    // baseReward=10, prev=0.5, new=0.8, gamma=0.99
    // F = 0.99·0.8 - 0.5 = 0.292 → total 10.292
    expect(potentialShaping(10, 0.5, 0.8, 0.99)).toBeCloseTo(10.292, 5);
  });

  it('returns base reward when potentials are equal', () => {
    expect(potentialShaping(5, 0.3, 0.3, 0.99)).toBeCloseTo(5, 1);
  });

  it('handles zero potential (pure base reward)', () => {
    expect(potentialShaping(5, 0, 0, 0.99)).toBeCloseTo(5, 10);
  });
});

describe('linearPotential', () => {
  it('scales progress by weight', () => {
    expect(linearPotential(0.5, 10)).toBe(5);
    expect(linearPotential(0, 10)).toBe(0);
    expect(linearPotential(1, 10)).toBe(10);
  });

  it('defaults to weight=1', () => {
    expect(linearPotential(0.7)).toBe(0.7);
  });
});

describe('progressShapingReward', () => {
  it('rewards progress gains', () => {
    // prev=0.5, new=0.8, gamma=0.99, weight=10
    // = 0.99·0.8·10 - 0.5·10 = 7.92 - 5 = 2.92
    expect(progressShapingReward(0.5, 0.8, 10, 0.99)).toBeCloseTo(2.92, 5);
  });

  it('zero when no progress', () => {
    expect(progressShapingReward(0.5, 0.5, 10, 0.99)).toBeCloseTo(0, 1);
  });

  it('negative when regressing', () => {
    expect(progressShapingReward(0.8, 0.5, 10, 0.99)).toBeLessThan(0);
  });
});

describe('Transition type (compile-time check)', () => {
  it('accepts custom state and action types', () => {
    // This is a type-only check; if it compiles, the types are sound.
    type CartPoleState = Float32Array;
    type CartPoleAction = number;
    const transition: Transition<CartPoleState, CartPoleAction> = {
      prevState: new Float32Array(4),
      action: 0,
      nextState: new Float32Array(4),
      reward: 1.0,
      done: false
    };
    expect(transition.action).toBe(0);
    expect(transition.reward).toBe(1.0);
  });
});
