/**
 * Q-Network interface — abstraction over the function approximator used by DQN.
 *
 * The DQN algorithm only cares about two operations:
 *   1. forward(state) → Q-values (one per action)
 *   2. train on a batch of (state, target_q) pairs, return the loss
 *
 * Implementations can be MLPs (default), dueling networks, or anything that
 * accepts a state batch and returns one Q-value per action.
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Forward + train contract for any Q-function approximator.
 */
export interface QNetwork {
  /**
   * Predict Q-values for a batch of states.
   * @param states - tf.Tensor2D of shape [batch, stateDim]
   * @returns tf.Tensor2D of shape [batch, actionDim]
   */
  predict(states: tf.Tensor2D): tf.Tensor2D;

  /**
   * Compute MSE loss between predicted Q-values and targets, then apply gradients.
   * @param states - [batch, stateDim]
   * @param targets - [batch, actionDim] (typically: predicted Q with target action replaced)
   * @returns The scalar loss value (for logging)
   */
  trainStep(states: tf.Tensor2D, targets: tf.Tensor2D): Promise<number>;

  /**
   * Copy weights from another Q-network. Used to update the target network.
   */
  copyWeightsFrom(other: QNetwork): Promise<void>;

  /**
   * Free all tensors and variables. Call on agent disposal.
   */
  dispose(): void;
}

/**
 * Configuration for the MLP Q-network.
 */
export interface MLPConfig {
  stateDim: number;
  actionDim: number;
  hiddenLayers?: number[]; // e.g. [128, 64]
  learningRate?: number; // Adam
  /**
   * Reserved for future use. The LayersModel API does not expose gradient
   * clipping directly; if you need it, use a custom QNetwork implementation
   * that uses `tf.variableGrads` + manual clipping.
   */
  gradientClipValue?: number;
}

/**
 * Simple multi-layer perceptron Q-network.
 * Hidden layers default to [128, 64] (Mnih et al. 2015, scaled down).
 */
export class MLPQNetwork implements QNetwork {
  public readonly stateDim: number;
  public readonly actionDim: number;
  public readonly hiddenLayers: number[];
  public readonly learningRate: number;
  public readonly gradientClipValue: number;

  public model: tf.LayersModel;

  constructor(config: MLPConfig) {
    this.stateDim = config.stateDim;
    this.actionDim = config.actionDim;
    this.hiddenLayers = config.hiddenLayers ?? [128, 64];
    this.learningRate = config.learningRate ?? 0.001;
    this.gradientClipValue = config.gradientClipValue ?? 1.0;

    this.model = this.buildModel();
  }

  private buildModel(): tf.LayersModel {
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [this.stateDim],
        units: this.hiddenLayers[0]!,
        activation: 'relu',
        kernelInitializer: 'heUniform'
      })
    );
    for (let i = 1; i < this.hiddenLayers.length; i++) {
      model.add(
        tf.layers.dense({
          units: this.hiddenLayers[i]!,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        })
      );
    }
    model.add(
      tf.layers.dense({
        units: this.actionDim,
        activation: 'linear',
        kernelInitializer: 'heUniform'
      })
    );
    return model;
  }

  predict(states: tf.Tensor2D): tf.Tensor2D {
    return this.model.predict(states) as tf.Tensor2D;
  }

  async trainStep(states: tf.Tensor2D, targets: tf.Tensor2D): Promise<number> {
    // Use the LayersModel optimizer API: we compile the model with an optimizer
    // and call `model.trainOnBatch` to do the gradient step.
    // This is the supported, type-safe path in TF.js for sequential models.

    // First, ensure the model has a compiled optimizer.
    if (!(this.model as unknown as { isOptimizerCompiled?: boolean }).isOptimizerCompiled) {
      this.model.compile({
        optimizer: tf.train.adam(this.learningRate),
        loss: tf.losses.meanSquaredError
      });
    }

    // trainOnBatch returns a History-like object with .loss
    const history = await this.model.trainOnBatch(states, targets);
    // history can be a number, an array, or a History object depending on TF version
    let loss: number;
    if (typeof history === 'number') {
      loss = history;
    } else if (Array.isArray(history)) {
      loss = (history as number[])[0]!;
    } else {
      const h = history as { loss: number | number[] };
      loss = Array.isArray(h.loss) ? h.loss[0]! : h.loss;
    }
    return loss;
  }

  async copyWeightsFrom(other: QNetwork): Promise<void> {
    if (!(other instanceof MLPQNetwork)) {
      throw new Error('copyWeightsFrom currently only supports MLPQNetwork → MLPQNetwork');
    }
    if (other.stateDim !== this.stateDim || other.actionDim !== this.actionDim) {
      throw new Error(
        `Network shape mismatch: target ${this.stateDim}x${this.actionDim}, ` +
          `source ${other.stateDim}x${other.actionDim}`
      );
    }
    // Use TF.js LayersModel.setWeights — much more efficient than dataSync round-trip
    this.model.setWeights(other.model.getWeights());
  }

  dispose(): void {
    this.model.dispose();
  }
}
