/**
 * DQN Agent (Deep Q-Network)
 *
 * Implementation of Mnih et al. 2015 ("Human-level control through deep reinforcement
 * learning") with the standard stabilization tricks:
 *   - Experience replay (random sampling from a buffer)
 *   - Target network (frozen copy, periodically synced)
 *   - Epsilon-greedy exploration with exponential decay
 *   - Gradient clipping on the loss
 *
 * Designed to be generic over the Q-network implementation: the default is
 * `MLPQNetwork` (a small MLP), but you can swap in a dueling network, conv-net,
 * or any other Q-function approximator that satisfies the `QNetwork` interface.
 *
 * Provenance: this is a re-implementation of the algorithm in oliverlaudan-ops/AI-Idle's
 * `src/systems/rl-bot/dqn-agent.js`, but with:
 *   - A typed QNetwork abstraction (so the network can be replaced without touching the agent)
 *   - Cleaner separation between exploration policy, training step, and target sync
 *   - Proper tensor disposal (no GPU memory leaks in long training runs)
 */

import * as tf from '@tensorflow/tfjs';
import type { QNetwork } from './q-network.js';
import { MLPQNetwork } from './q-network.js';
import { ReplayBuffer } from '../core/replay-buffer.js';

export interface DQNAgentConfig<S = Float32Array> {
  /** Number of discrete actions in the environment. */
  actionDim: number;
  /** State vector dimension. */
  stateDim: number;
  /** Discount factor (default 0.99). */
  gamma?: number;
  /** Replay buffer capacity (default 10_000). */
  bufferCapacity?: number;
  /** Mini-batch size per training step (default 32). */
  batchSize?: number;
  /** Initial epsilon (default 1.0 — pure exploration). */
  epsilonStart?: number;
  /** Final epsilon (default 0.01). */
  epsilonEnd?: number;
  /** Epsilon decay per training step (default 0.995). */
  epsilonDecay?: number;
  /** Update target network every N training steps (default 100). */
  targetUpdateFreq?: number;
  /** Hidden layer sizes (default [128, 64]). */
  hiddenLayers?: number[];
  /** Learning rate (default 0.001). */
  learningRate?: number;
  /** Gradient clip value, 0 to disable (default 1.0). */
  gradientClipValue?: number;
  /** Optional custom Q-network. Defaults to MLPQNetwork. */
  qNetworkFactory?: () => QNetwork;
  /** Optional buffer override (e.g., PrioritizedReplayBuffer). */
  buffer?: ReplayBuffer<S>;
}

export interface DQNTrainingStats {
  loss: number;
  meanQ: number;
  epsilon: number;
  bufferSize: number;
  steps: number;
}

/**
 * The DQN agent. Manages online + target networks, replay buffer, and training.
 */
export class DQNAgent<S = Float32Array> {
  public readonly actionDim: number;
  public readonly stateDim: number;
  public readonly gamma: number;
  public readonly batchSize: number;
  public readonly epsilonStart: number;
  public readonly epsilonEnd: number;
  public readonly epsilonDecay: number;
  public readonly targetUpdateFreq: number;

  public readonly online: QNetwork;
  public readonly target: QNetwork;
  public readonly buffer: ReplayBuffer<S>;

  protected epsilon: number;
  protected steps = 0;
  protected totalSteps = 0;
  protected lastLoss = NaN;
  protected lastMeanQ = NaN;

  constructor(config: DQNAgentConfig<S>) {
    this.actionDim = config.actionDim;
    this.stateDim = config.stateDim;
    this.gamma = config.gamma ?? 0.99;
    this.batchSize = config.batchSize ?? 32;
    this.epsilonStart = config.epsilonStart ?? 1.0;
    this.epsilonEnd = config.epsilonEnd ?? 0.01;
    this.epsilonDecay = config.epsilonDecay ?? 0.995;
    this.targetUpdateFreq = config.targetUpdateFreq ?? 100;

    // Networks
    const factory = config.qNetworkFactory ?? ((): QNetwork => new MLPQNetwork({
      stateDim: this.stateDim,
      actionDim: this.actionDim,
      hiddenLayers: config.hiddenLayers,
      learningRate: config.learningRate,
      gradientClipValue: config.gradientClipValue
    }));
    this.online = factory();
    this.target = factory();
    // Sync target with online initially
    void this.target.copyWeightsFrom(this.online);

    // Buffer
    this.buffer = config.buffer ?? new ReplayBuffer<S>(config.bufferCapacity ?? 10_000);

    this.epsilon = this.epsilonStart;
  }

  /**
   * Select an action using ε-greedy policy.
   * During training, with probability ε the agent picks a random action;
   * otherwise it picks argmax_a Q(s, a).
   * @param state - Current state observation
   * @param training - If false, no exploration (greedy). Default true.
   */
  async selectAction(state: S, training = true): Promise<number> {
    if (training && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionDim);
    }
    return tf.tidy((): number => {
      const stateBatch = tf.tensor2d(Array.from(state as unknown as ArrayLike<number>), [
        1,
        this.stateDim
      ]);
      const qValues = this.online.predict(stateBatch);
      const qData = qValues.dataSync();
      let bestAction = 0;
      let bestQ = qData[0]!;
      for (let a = 1; a < this.actionDim; a++) {
        if (qData[a]! > bestQ) {
          bestQ = qData[a]!;
          bestAction = a;
        }
      }
      return bestAction;
    });
  }

  /**
   * Add an experience to the replay buffer.
   */
  remember(state: S, action: number, reward: number, nextState: S, done: boolean): void {
    this.buffer.add(state, action, reward, nextState, done);
  }

  /**
   * Run one training step: sample a batch, compute the Bellman target, update online network.
   * Returns the loss (or NaN if the buffer doesn't have enough samples yet).
   */
  async trainStep(): Promise<number> {
    if (this.buffer.length() < this.batchSize) {
      return NaN;
    }

    const batch = this.buffer.sample(this.batchSize);
    const statesArr = batch.states.map((s) => Array.from(s as unknown as ArrayLike<number>));
    const nextStatesArr = batch.nextStates.map((s) =>
      Array.from(s as unknown as ArrayLike<number>)
    );

    // Build target Q-matrix: y_i = r_i + γ * max_a Q_target(s', a) * (1 - done_i)
    let meanQ = NaN;
    const targets = tf.tidy((): tf.Tensor2D => {
      const statesT = tf.tensor2d(statesArr, [this.batchSize, this.stateDim]);
      const nextStatesT = tf.tensor2d(nextStatesArr, [this.batchSize, this.stateDim]);

      const qNext = this.target.predict(nextStatesT);
      const maxQNext = qNext.max(1) as tf.Tensor1D;
      // Track mean Q from online network for logging
      const qPred = this.online.predict(statesT);
      meanQ = qPred.mean().dataSync()[0]!;

      const rewardsT = tf.tensor1d(batch.rewards);
      const donesT = tf.tensor1d(batch.dones);
      const actionsT = tf.tensor1d(batch.actions, 'int32');

      const bootstrap = maxQNext.mul(tf.scalar(1).sub(donesT));
      const targetScalar = rewardsT.add(bootstrap.mul(this.gamma));

      const oneHot = tf.oneHot(actionsT, this.actionDim) as tf.Tensor2D;
      return oneHot.mul(targetScalar.reshape([this.batchSize, 1])) as tf.Tensor2D;
    });

    const statesT = tf.tensor2d(statesArr, [this.batchSize, this.stateDim]);
    const loss = await this.online.trainStep(statesT, targets);
    statesT.dispose();
    targets.dispose();

    this.lastLoss = loss;
    this.lastMeanQ = meanQ;
    this.steps++;
    this.totalSteps++;

    // Decay epsilon
    this.epsilon = Math.max(this.epsilonEnd, this.epsilon * this.epsilonDecay);

    // Sync target network periodically
    if (this.steps % this.targetUpdateFreq === 0) {
      await this.target.copyWeightsFrom(this.online);
    }

    return loss;
  }

  /**
   * Get current training statistics.
   */
  getStats(): DQNTrainingStats {
    return {
      loss: this.lastLoss,
      meanQ: this.lastMeanQ,
      epsilon: this.epsilon,
      bufferSize: this.buffer.length(),
      steps: this.totalSteps
    };
  }

  /**
   * Force a target network sync.
   */
  async syncTarget(): Promise<void> {
    await this.target.copyWeightsFrom(this.online);
  }

  /**
   * Free all TF.js resources. Call on agent disposal.
   */
  dispose(): void {
    this.online.dispose();
    this.target.dispose();
  }
}
