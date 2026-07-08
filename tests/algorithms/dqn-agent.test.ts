import { describe, it, expect, beforeAll, beforeEach, afterEach } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { DQNAgent } from '../../src/algorithms/dqn-agent.js';
import { CartPoleEnv } from '../../src/envs/cartpole.js';

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

describe('DQNAgent', () => {
  let agent: DQNAgent;

  beforeEach(() => {
    agent = new DQNAgent({
      actionDim: 2,
      stateDim: 4,
      bufferCapacity: 200,
      batchSize: 4,
      epsilonStart: 0.5,
      epsilonEnd: 0.1,
      epsilonDecay: 0.9,
      targetUpdateFreq: 2,
      learningRate: 0.01
    });
  });

  afterEach(() => {
    agent.dispose();
  });

  describe('constructor', () => {
    it('initializes with starting epsilon', () => {
      expect(agent.epsilon).toBe(0.5);
    });

    it('creates online and target networks with correct dimensions', () => {
      expect(agent.online).toBeDefined();
      expect(agent.target).toBeDefined();
      // Both should be the same architecture (4 → 128 → 64 → 2)
      const inShape = agent.online.model.inputs[0]!.shape;
      const outShape = agent.online.model.outputs[0]!.shape;
      expect(inShape).toEqual([null, 4]);
      expect(outShape).toEqual([null, 2]);
    });

    it('uses default hyperparameters', () => {
      const defaultAgent = new DQNAgent({ actionDim: 2, stateDim: 4 });
      expect(defaultAgent.gamma).toBe(0.99);
      expect(defaultAgent.batchSize).toBe(32);
      expect(defaultAgent.epsilonStart).toBe(1.0);
      expect(defaultAgent.epsilonEnd).toBe(0.01);
      expect(defaultAgent.epsilonDecay).toBe(0.995);
      expect(defaultAgent.targetUpdateFreq).toBe(100);
      defaultAgent.dispose();
    });
  });

  describe('selectAction()', () => {
    it('returns a valid action in [0, actionDim)', async () => {
      const state = new Float32Array([0, 0, 0, 0]);
      for (let i = 0; i < 20; i++) {
        const a = await agent.selectAction(state, false);
        expect(a).toBeGreaterThanOrEqual(0);
        expect(a).toBeLessThan(2);
      }
    });

    it('explores randomly with high epsilon', async () => {
      // With epsilon=0.5, expect both actions to be seen
      const state = new Float32Array([0, 0, 0, 0]);
      const seen = new Set<number>();
      for (let i = 0; i < 50; i++) {
        seen.add(await agent.selectAction(state, true));
      }
      expect(seen.size).toBe(2);
    });

    it('is greedy when training=false', async () => {
      // All states are zero — initially Q-values are random. Greedy should still
      // return a valid action and be deterministic across calls.
      const state = new Float32Array([0, 0, 0, 0]);
      const first = await agent.selectAction(state, false);
      const second = await agent.selectAction(state, false);
      expect(first).toBe(second);
    });

    it('respects actionDim', async () => {
      const envAgent = new DQNAgent({ actionDim: 5, stateDim: 4 });
      const state = new Float32Array([0, 0, 0, 0]);
      for (let i = 0; i < 20; i++) {
        const a = await envAgent.selectAction(state, false);
        expect(a).toBeLessThan(5);
      }
      envAgent.dispose();
    });
  });

  describe('remember()', () => {
    it('adds experience to the buffer', () => {
      expect(agent.buffer.length()).toBe(0);
      agent.remember(new Float32Array([0, 0, 0, 0]), 0, 1, new Float32Array([0, 0, 0, 0]), false);
      expect(agent.buffer.length()).toBe(1);
    });
  });

  describe('trainStep()', () => {
    it('returns NaN when buffer has fewer experiences than batchSize', async () => {
      agent.remember(new Float32Array([0, 0, 0, 0]), 0, 1, new Float32Array([0, 0, 0, 0]), false);
      agent.remember(new Float32Array([0, 0, 0, 0]), 1, 1, new Float32Array([0, 0, 0, 0]), false);
      const loss = await agent.trainStep();
      expect(Number.isNaN(loss)).toBe(true);
    });

    it('returns a finite loss when buffer has enough experiences', async () => {
      for (let i = 0; i < 10; i++) {
        agent.remember(
          new Float32Array([i, i, i, i]),
          i % 2,
          1.0,
          new Float32Array([i + 1, i, i, i]),
          false
        );
      }
      const loss = await agent.trainStep();
      expect(Number.isFinite(loss)).toBe(true);
    });

    it('decays epsilon after training step', async () => {
      const initialEpsilon = agent.epsilon;
      for (let i = 0; i < 10; i++) {
        agent.remember(
          new Float32Array([i, i, i, i]),
          i % 2,
          1.0,
          new Float32Array([i + 1, i, i, i]),
          false
        );
      }
      await agent.trainStep();
      expect(agent.epsilon).toBeLessThan(initialEpsilon);
    });

    it('clamps epsilon at epsilonEnd', async () => {
      // After many steps, epsilon should be capped
      for (let i = 0; i < 100; i++) {
        agent.remember(
          new Float32Array([i, i, i, i]),
          i % 2,
          1.0,
          new Float32Array([i + 1, i, i, i]),
          false
        );
      }
      for (let i = 0; i < 50; i++) {
        await agent.trainStep();
      }
      expect(agent.epsilon).toBeCloseTo(agent.epsilonEnd, 5);
    });

    it('increments step counter', async () => {
      for (let i = 0; i < 10; i++) {
        agent.remember(
          new Float32Array([i, i, i, i]),
          i % 2,
          1.0,
          new Float32Array([i + 1, i, i, i]),
          false
        );
      }
      const stats0 = agent.getStats();
      await agent.trainStep();
      const stats1 = agent.getStats();
      expect(stats1.steps).toBe(stats0.steps + 1);
    });
  });

  describe('getStats()', () => {
    it('returns current statistics', async () => {
      const stats = agent.getStats();
      expect(stats.epsilon).toBe(0.5);
      expect(stats.bufferSize).toBe(0);
      expect(stats.steps).toBe(0);
      expect(Number.isNaN(stats.loss)).toBe(true);
    });

    it('updates after training', async () => {
      for (let i = 0; i < 10; i++) {
        agent.remember(
          new Float32Array([i, i, i, i]),
          i % 2,
          1.0,
          new Float32Array([i + 1, i, i, i]),
          false
        );
      }
      await agent.trainStep();
      const stats = agent.getStats();
      expect(stats.bufferSize).toBe(10);
      expect(stats.steps).toBe(1);
      expect(Number.isFinite(stats.loss)).toBe(true);
    });
  });

  describe('syncTarget()', () => {
    it('copies weights from online to target', async () => {
      // Train online to change weights
      for (let i = 0; i < 10; i++) {
        agent.remember(
          new Float32Array([i, i, i, i]),
          i % 2,
          1.0,
          new Float32Array([i + 1, i, i, i]),
          false
        );
      }
      for (let i = 0; i < 5; i++) {
        await agent.trainStep();
      }
      // Now sync
      await agent.syncTarget();
      // Both should produce the same output
      const state = tf.tensor2d([[1, 2, 3, 4]], [1, 4]);
      const onlineQ = agent.online.predict(state).dataSync();
      const targetQ = agent.target.predict(state).dataSync();
      for (let i = 0; i < onlineQ.length; i++) {
        expect(targetQ[i]).toBeCloseTo(onlineQ[i]!, 5);
      }
      state.dispose();
    });
  });

  describe('integration with CartPole', () => {
    it('completes 10 full episodes without errors', async () => {
      const env = new CartPoleEnv({ maxSteps: 50 });
      for (let episode = 0; episode < 10; episode++) {
        let state = env.reset();
        let done = false;
        let steps = 0;
        while (!done && steps < 50) {
          const action = await agent.selectAction(state, true);
          const result = env.step(action);
          agent.remember(state, action, result.reward, result.state, result.done);
          if (agent.buffer.length() >= agent.batchSize) {
            await agent.trainStep();
          }
          state = result.state;
          done = result.done;
          steps++;
        }
      }
      const stats = agent.getStats();
      expect(stats.steps).toBeGreaterThan(0);
    }, 60_000);
  });
});
