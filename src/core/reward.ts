/**
 * Reward Function Utilities
 *
 * Generic reward-calculation utilities for reinforcement learning.
 * The actual reward logic is environment-specific — this module provides
 * the math (return, normalization, statistics) and a `RewardFn` type
 * that callers implement.
 *
 * Design: avoid a monolithic switch-case. Each environment plugs in its
 * own `RewardFn<State, Action>` that knows what the state and action mean.
 */

/**
 * A state transition: what the agent saw before, what it did, and what it saw after.
 * Both states and action are generic — the environment decides their types.
 *
 * @typeParam S - State type
 * @typeParam A - Action type (typically number for discrete, Float32Array for continuous)
 */
export interface Transition<S = Float32Array, A = number> {
  prevState: S;
  action: A;
  nextState: S;
  reward: number;
  done: boolean;
}

/**
 * A reward function maps a transition to a scalar reward.
 * Implementations are environment-specific.
 */
export type RewardFn<S = Float32Array, A = number> = (transition: Transition<S, A>) => number;

/**
 * Compute the discounted return from a sequence of rewards.
 * G_t = r_t + γ * G_{t+1}, computed backward.
 *
 * @param rewards - Sequence of rewards from an episode
 * @param gamma - Discount factor, default 0.99
 * @returns The discounted return from the first reward in the array
 *
 * Note: with gamma=0, this returns only the first reward (degenerate case).
 */
export function calculateReturn(rewards: number[], gamma = 0.99): number {
  let G = 0;
  for (let t = rewards.length - 1; t >= 0; t--) {
    G = rewards[t]! + gamma * G;
  }
  return G;
}

/**
 * Compute the n-step discounted return from a sequence of rewards.
 * Useful for n-step TD learning.
 */
export function calculateNStepReturn(rewards: number[], n: number, gamma = 0.99): number {
  let G = 0;
  const limit = Math.min(n, rewards.length);
  for (let t = limit - 1; t >= 0; t--) {
    G = rewards[t]! + gamma * G;
  }
  return G;
}

/**
 * Normalize rewards to zero mean and unit standard deviation.
 * Helps with training stability.
 *
 * @param rewards - Array of rewards
 * @returns Normalized rewards with same length
 */
export function normalizeRewards(rewards: number[]): number[] {
  if (rewards.length === 0) return [];

  const mean = rewards.reduce((s, r) => s + r, 0) / rewards.length;
  const variance = rewards.reduce((s, r) => s + (r - mean) ** 2, 0) / rewards.length;
  const std = Math.sqrt(variance) + 1e-8; // epsilon prevents division by zero

  return rewards.map((r) => (r - mean) / std);
}

/**
 * Clip rewards to [-maxAbs, maxAbs]. Useful for additional training stability
 * when raw rewards have extreme outliers (e.g., sparse large-reward environments).
 */
export function clipRewards(rewards: number[], maxAbs: number): number[] {
  return rewards.map((r) => Math.max(-maxAbs, Math.min(maxAbs, r)));
}

/**
 * Reward statistics for an episode.
 */
export interface RewardStats {
  total: number;
  mean: number;
  min: number;
  max: number;
  std: number;
  count: number;
}

/**
 * Compute statistics for a reward sequence.
 */
export function getRewardStats(rewards: number[]): RewardStats {
  if (rewards.length === 0) {
    return { total: 0, mean: 0, min: 0, max: 0, std: 0, count: 0 };
  }
  const total = rewards.reduce((s, r) => s + r, 0);
  const mean = total / rewards.length;
  const min = Math.min(...rewards);
  const max = Math.max(...rewards);
  const variance = rewards.reduce((s, r) => s + (r - mean) ** 2, 0) / rewards.length;
  const std = Math.sqrt(variance);
  return { total, mean, min, max, std, count: rewards.length };
}

/**
 * Generalized Advantage Estimation (GAE).
 * Used by policy-gradient methods (PPO, A2C).
 *
 * A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
 * where δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
 *
 * @param rewards - Rewards from t=0 to t=T-1
 * @param values - State values V(s_t) from t=0 to t=T (length = rewards.length + 1)
 * @param gamma - Discount factor, default 0.99
 * @param lambda - GAE smoothing parameter, default 0.95
 * @returns Advantages for t=0 to t=T-1
 */
export function calculateAdvantages(
  rewards: number[],
  values: number[],
  gamma = 0.99,
  lambda = 0.95
): number[] {
  const advantages: number[] = [];
  let lastAdvantage = 0;
  const T = rewards.length;
  for (let t = T - 1; t >= 0; t--) {
    const v_t = values[t] ?? 0;
    const v_next = values[t + 1] ?? 0;
    const delta = rewards[t]! + gamma * v_next - v_t;
    lastAdvantage = delta + gamma * lambda * lastAdvantage;
    advantages.unshift(lastAdvantage);
  }
  return advantages;
}

/**
 * Apply potential-based reward shaping.
 *
 * F(s, s') = γ * Φ(s') - Φ(s)
 *
 * Preserves optimal policies when the potential function is bounded
 * (Ng et al. 1999). Add F to the base reward to speed up learning
 * without changing what the agent is trying to achieve.
 *
 * @param baseReward - Original reward
 * @param prevPotential - Potential of previous state
 * @param newPotential - Potential of new state
 * @param gamma - Discount factor
 */
export function potentialShaping(
  baseReward: number,
  prevPotential: number,
  newPotential: number,
  gamma = 0.99
): number {
  return baseReward + gamma * newPotential - prevPotential;
}

/**
 * Linear potential function for use with `potentialShaping`.
 * Maps a scalar "progress" value (0..1) to a potential.
 */
export const linearPotential = (progress: number, weight = 1): number => progress * weight;

/**
 * Convenience: compute a "potential difference" reward given two progress values.
 * Useful when the env exposes a `progress` field (e.g., % to goal).
 */
export function progressShapingReward(
  prevProgress: number,
  newProgress: number,
  weight = 1,
  gamma = 0.99
): number {
  // Φ(s) = progress(s) * weight;  Δ = γ * newPotential - prevPotential
  return gamma * newProgress * weight - prevProgress * weight;
}
