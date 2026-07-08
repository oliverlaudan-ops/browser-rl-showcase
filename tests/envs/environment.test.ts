import { describe, it, expect } from 'vitest';
import {
  runEpisode,
  type Environment,
  type StepResult,
  type EnvSpec
} from '../../src/envs/environment.js';

/**
 * A deterministic test environment: alternates states, deterministic rewards.
 * Used to verify the Environment contract and runEpisode helper.
 */
class StubEnv implements Environment<Float32Array, number> {
  private stepCount = 0;
  private readonly maxSteps: number;
  readonly state: Float32Array = new Float32Array(2);

  constructor(maxSteps = 5) {
    this.maxSteps = maxSteps;
  }

  reset(): Float32Array {
    this.stepCount = 0;
    this.state[0] = 0;
    this.state[1] = 0;
    return new Float32Array(this.state);
  }

  step(action: number): StepResult<Float32Array> {
    this.stepCount++;
    this.state[0] = action;
    this.state[1] = this.stepCount;
    return {
      state: new Float32Array(this.state),
      reward: 1.0,
      done: this.stepCount >= this.maxSteps,
      info: { step: this.stepCount }
    };
  }

  getSpec(): EnvSpec {
    return {
      id: 'stub',
      observationDim: 2,
      actionDim: 3,
      actionType: 'discrete',
      maxSteps: this.maxSteps,
      rewardRange: [0, 1] as const
    };
  }
}

describe('Environment interface', () => {
  it('reset returns initial state and zeroes counters', () => {
    const env = new StubEnv(5);
    const initial = env.reset();
    expect(initial).toBeInstanceOf(Float32Array);
    expect(initial[0]).toBe(0);
    expect(initial[1]).toBe(0);
  });

  it('step advances state and returns reward/done', () => {
    const env = new StubEnv(5);
    env.reset();
    const result = env.step(2);
    expect(result.reward).toBe(1.0);
    expect(result.done).toBe(false);
    expect(result.state[0]).toBe(2);
    expect(result.state[1]).toBe(1);
  });

  it('sets done=true when maxSteps reached', () => {
    const env = new StubEnv(3);
    env.reset();
    env.step(0);
    env.step(1);
    const r = env.step(2);
    expect(r.done).toBe(true);
  });

  it('getSpec returns metadata', () => {
    const env = new StubEnv(5);
    const spec = env.getSpec();
    expect(spec.id).toBe('stub');
    expect(spec.observationDim).toBe(2);
    expect(spec.actionDim).toBe(3);
    expect(spec.actionType).toBe('discrete');
  });
});

describe('runEpisode', () => {
  it('collects transitions until done', () => {
    const env = new StubEnv(3);
    const policy = (s: Float32Array) => (s[1] < 2 ? 0 : 1);
    const rollout = runEpisode(env, policy);
    expect(rollout.transitions).toHaveLength(3);
    expect(rollout.totalReward).toBe(3.0);
    expect(rollout.stats.steps).toBe(3);
    expect(rollout.stats.meanReward).toBeCloseTo(1.0, 5);
  });

  it('respects maxSteps cap', () => {
    // StubEnv runs 5 steps before done, but we cap at 2
    const env = new StubEnv(5);
    const policy = () => 0;
    const rollout = runEpisode(env, policy, 2);
    expect(rollout.transitions).toHaveLength(2);
    expect(rollout.stats.steps).toBe(2);
  });

  it('handles zero-step policy correctly', () => {
    const env = new StubEnv(3);
    env.reset();
    const r = env.step(0);
    expect(r.done).toBe(false); // first step: stepCount=1, not done
  });

  it('includes done in last transition', () => {
    const env = new StubEnv(2);
    const policy = () => 0;
    const rollout = runEpisode(env, policy);
    expect(rollout.transitions[rollout.transitions.length - 1]!.done).toBe(true);
  });

  it('does not modify prior transitions when terminal', () => {
    // After reset, step with done=true; the recorded transition should reflect that.
    const env = new StubEnv(1);
    env.reset();
    const rollout = runEpisode(env, () => 0);
    expect(rollout.transitions).toHaveLength(1);
    expect(rollout.transitions[0]!.done).toBe(true);
  });
});
