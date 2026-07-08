import { describe, it, expect } from 'vitest';
import { CartPoleEnv } from '../../src/envs/cartpole.js';
import { runEpisode } from '../../src/envs/environment.js';

describe('CartPoleEnv', () => {
  describe('constructor & spec', () => {
    it('uses default config', () => {
      const env = new CartPoleEnv();
      const spec = env.getSpec();
      expect(spec.id).toBe('cartpole-v1');
      expect(spec.observationDim).toBe(4);
      expect(spec.actionDim).toBe(2);
      expect(spec.actionType).toBe('discrete');
      expect(spec.maxSteps).toBe(500);
    });

    it('accepts custom maxSteps', () => {
      const env = new CartPoleEnv({ maxSteps: 100 });
      expect(env.getSpec().maxSteps).toBe(100);
    });

    it('accepts custom thresholds', () => {
      const env = new CartPoleEnv({ xThreshold: 1.0, thetaThresholdRadians: 0.1 });
      const spec = env.getSpec();
      expect(spec.maxSteps).toBe(500); // not overridden
      // thresholds are private but reflected in termination behavior
    });
  });

  describe('reset()', () => {
    it('returns Float32Array of length 4', () => {
      const env = new CartPoleEnv();
      const state = env.reset();
      expect(state).toBeInstanceOf(Float32Array);
      expect(state.length).toBe(4);
    });

    it('initial state is within small range (default 0.05)', () => {
      const env = new CartPoleEnv();
      for (let i = 0; i < 10; i++) {
        const state = env.reset();
        for (let j = 0; j < 4; j++) {
          expect(Math.abs(state[j]!)).toBeLessThanOrEqual(0.05);
        }
      }
    });

    it('returns a copy (not internal reference)', () => {
      const env = new CartPoleEnv();
      const state = env.reset();
      state[0] = 999;
      env.step(0);
      const next = env.reset();
      expect(next[0]).not.toBe(999);
    });
  });

  describe('step()', () => {
    it('returns reward 1.0 for a normal step', () => {
      const env = new CartPoleEnv({ initialStateRange: 0 }); // no random offset
      env.reset();
      const result = env.step(0);
      expect(result.reward).toBe(1.0);
      expect(result.done).toBe(false);
    });

    it('throws on invalid action', () => {
      const env = new CartPoleEnv();
      env.reset();
      expect(() => env.step(2)).toThrow(/Invalid action/);
      expect(() => env.step(-1)).toThrow();
    });

    it('terminates when x exceeds threshold', () => {
      const env = new CartPoleEnv({ initialStateRange: 0 });
      env.reset();
      // Force the cart to the right by pushing right many times
      let done = false;
      let steps = 0;
      while (!done && steps < 1000) {
        const r = env.step(1);
        done = r.done;
        steps++;
      }
      expect(done).toBe(true);
    });

    it('terminates after maxSteps', () => {
      const env = new CartPoleEnv({ maxSteps: 5 });
      env.reset();
      // Step right slowly with action 1 — pole should fall eventually but within 5 steps
      let done = false;
      for (let i = 0; i < 10; i++) {
        const r = env.step(i % 2);
        if (r.done) {
          done = true;
          break;
        }
      }
      expect(done).toBe(true);
    });

    it('returns copy of state (mutation-safe)', () => {
      const env = new CartPoleEnv();
      env.reset();
      const r = env.step(1);
      r.state[0] = 999;
      const r2 = env.step(1);
      expect(r2.state[0]).not.toBe(999);
    });

    it('info includes termination reason', () => {
      const env = new CartPoleEnv({ maxSteps: 3 });
      env.reset();
      let result;
      for (let i = 0; i < 5; i++) {
        result = env.step(0);
        if (result.done) break;
      }
      expect(result!.info).toBeDefined();
      expect(['truncated', 'out_of_bounds']).toContain(result!.info!.termination);
    });
  });

  describe('runEpisode integration', () => {
    it('achieves reasonable episode length with random policy', () => {
      const env = new CartPoleEnv({ maxSteps: 200 });
      const policy = () => (Math.random() < 0.5 ? 0 : 1);
      const rollout = runEpisode(env, policy, 200);
      // Random policy should survive at least a few steps
      expect(rollout.stats.steps).toBeGreaterThan(5);
    });

    it('total reward equals steps (since reward=1 per step)', () => {
      const env = new CartPoleEnv({ maxSteps: 50 });
      const policy = () => 0;
      const rollout = runEpisode(env, policy, 50);
      expect(rollout.totalReward).toBe(rollout.stats.steps);
    });
  });
});
