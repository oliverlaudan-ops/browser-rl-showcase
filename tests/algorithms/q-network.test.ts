import { describe, it, expect, beforeAll } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { MLPQNetwork } from '../../src/algorithms/q-network.js';

// CPU backend is the most reliable in Node — no GPU, no WebGL, deterministic.
beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

describe('MLPQNetwork', () => {
  describe('constructor', () => {
    it('creates a model with correct input/output shapes', () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 2 });
      const inputShape = net.model.inputs[0]!.shape;
      const outputShape = net.model.outputs[0]!.shape;
      expect(inputShape).toEqual([null, 4]);
      expect(outputShape).toEqual([null, 2]);
    });

    it('uses default hidden layers [128, 64] when none specified', () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 2 });
      expect(net.hiddenLayers).toEqual([128, 64]);
    });

    it('accepts custom hidden layers', () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 2, hiddenLayers: [32, 16] });
      expect(net.hiddenLayers).toEqual([32, 16]);
    });

    it('uses default learning rate 0.001', () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 2 });
      expect(net.learningRate).toBe(0.001);
    });

    it('accepts custom learning rate', () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 2, learningRate: 0.01 });
      expect(net.learningRate).toBe(0.01);
    });
  });

  describe('predict()', () => {
    it('returns Q-values of shape [batch, actionDim]', () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 3 });
      const states = tf.tensor2d(
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0]
        ],
        [2, 4]
      );
      const qValues = net.predict(states);
      expect(qValues.shape).toEqual([2, 3]);
      qValues.dispose();
      states.dispose();
    });

    it('is deterministic (same input → same output)', () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 2 });
      const states = tf.tensor2d([[1, 2, 3, 4]], [1, 4]);
      const a1 = net.predict(states).dataSync();
      const a2 = net.predict(states).dataSync();
      expect(Array.from(a1)).toEqual(Array.from(a2));
      states.dispose();
    });
  });

  describe('trainStep()', () => {
    it('returns a finite loss value', async () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 2, learningRate: 0.01 });
      const states = tf.tensor2d(
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]
        ],
        [4, 4]
      );
      const targets = tf.tensor2d(
        [
          [1, 0],
          [0, 1],
          [1, 0],
          [0, 1]
        ],
        [4, 2]
      );
      const loss = await net.trainStep(states, targets);
      expect(Number.isFinite(loss)).toBe(true);
      expect(loss).toBeGreaterThanOrEqual(0);
      states.dispose();
      targets.dispose();
    });

    it('reduces loss over multiple training steps (gradient descent works)', async () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 2, learningRate: 0.1 });
      const states = tf.tensor2d([[1, 0, 0, 0]], [1, 4]);
      const targets = tf.tensor2d([[1, 0]], [1, 2]);
      const first = await net.trainStep(states, targets);
      const second = await net.trainStep(states, targets);
      const third = await net.trainStep(states, targets);
      // With a high learning rate, loss should decrease over a few steps
      // (not guaranteed to be strictly monotonic, but should trend down)
      expect(third).toBeLessThan(first);
      // For sanity: the difference shouldn't be negative by a huge amount
      expect(third).toBeGreaterThanOrEqual(0);
      states.dispose();
      targets.dispose();
    });
  });

  describe('copyWeightsFrom()', () => {
    it('copies weights from one network to another', async () => {
      const src = new MLPQNetwork({ stateDim: 4, actionDim: 2 });
      const dst = new MLPQNetwork({ stateDim: 4, actionDim: 2 });

      // Force src to produce a known output by training it briefly
      const states = tf.tensor2d([[1, 0, 0, 0]], [1, 4]);
      const targets = tf.tensor2d([[1, 0]], [1, 2]);
      await src.trainStep(states, targets);

      // Capture src's output
      const srcPred = src.predict(states).dataSync();

      // Copy weights to dst
      await dst.copyWeightsFrom(src);

      // dst should now produce the same output
      const dstPred = dst.predict(states).dataSync();

      for (let i = 0; i < srcPred.length; i++) {
        expect(dstPred[i]).toBeCloseTo(srcPred[i]!, 5);
      }

      states.dispose();
      targets.dispose();
    });

    it('throws on shape mismatch', async () => {
      const a = new MLPQNetwork({ stateDim: 4, actionDim: 2 });
      const b = new MLPQNetwork({ stateDim: 5, actionDim: 2 });
      await expect(a.copyWeightsFrom(b)).rejects.toThrow(/shape mismatch/i);
    });
  });

  describe('dispose()', () => {
    it('does not throw', () => {
      const net = new MLPQNetwork({ stateDim: 4, actionDim: 2 });
      expect(() => net.dispose()).not.toThrow();
    });
  });
});
