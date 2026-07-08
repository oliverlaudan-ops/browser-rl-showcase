/**
 * Environment Interface
 *
 * The contract between an RL agent and the world it learns from.
 * Any environment that implements this interface can be plugged into
 * the same training loop.
 *
 * Design goals:
 * - Pure interface — no assumption about how the environment is rendered
 *   or what its state representation looks like.
 * - Discrete and continuous action spaces supported via the union type.
 * - Reset returns a fresh state; step returns the next state and metadata.
 */

/**
 * Discrete action: an integer in [0, actionDim).
 * Continuous action: a Float32Array (e.g., for robotics).
 */
export type DiscreteAction = number;
export type ContinuousAction = Float32Array;
export type Action = DiscreteAction | ContinuousAction;

/**
 * What `step()` returns to the agent.
 *
 * Note: `reward` and `done` are *not* part of the state — they're metadata
 * about the transition.
 */
export interface StepResult<S = Float32Array> {
  state: S; // next state
  reward: number;
  done: boolean;
  info?: Record<string, unknown>; // environment-specific extras (e.g., episode length, win/loss)
}

/**
 * Environment metadata. Returned by `getSpec()`.
 */
export interface EnvSpec {
  readonly id: string; // unique env identifier
  readonly observationDim: number; // state vector length
  readonly actionDim: number; // number of discrete actions (1 if continuous)
  readonly actionType: 'discrete' | 'continuous';
  readonly maxSteps?: number; // episode step limit
  readonly rewardRange?: readonly [number, number]; // [min, max] reward per step
}

/**
 * The Environment contract.
 *
 * @typeParam S - State type (e.g., Float32Array for vector envs)
 * @typeParam A - Action type (number for discrete, Float32Array for continuous)
 */
export interface Environment<S = Float32Array, A extends Action = DiscreteAction> {
  /**
   * Reset the environment to its initial state.
   * Returns the initial state observation.
   */
  reset(): S;

  /**
   * Take one action in the environment.
   * Mutates internal state; returns the next state and transition metadata.
   */
  step(action: A): StepResult<S>;

  /**
   * Environment metadata for agent configuration.
   * Should be cheap to call (no side effects).
   */
  getSpec(): EnvSpec;

  /**
   * Optional: render the current state (for UI/visualization).
   * Default: no-op. Environments with rendering implement this.
   */
  render?(): void;

  /**
   * Optional: close the environment, release resources.
   */
  close?(): void;
}

/**
 * Episode statistics — collected during a rollout.
 */
export interface EpisodeStats {
  steps: number;
  totalReward: number;
  meanReward: number;
  length: number; // alias for steps, in case "length" reads better
}

/**
 * Run a complete episode in the environment using a policy.
 * Generic over state and action types.
 *
 * @param env - The environment
 * @param policy - Function that maps a state to an action
 * @param maxSteps - Hard cap on episode length (e.g., to bound infinite envs)
 * @returns The collected transitions, the total reward, and episode statistics
 */
export interface Rollout<S = Float32Array, A extends Action = DiscreteAction> {
  transitions: Array<{
    state: S;
    action: A;
    nextState: S;
    reward: number;
    done: boolean;
  }>;
  totalReward: number;
  stats: EpisodeStats;
}

/**
 * Run one episode with a policy.
 */
export function runEpisode<S, A extends Action>(
  env: Environment<S, A>,
  policy: (state: S) => A,
  maxSteps = 10_000
): Rollout<S, A> {
  const transitions: Rollout<S, A>['transitions'] = [];
  let state = env.reset();
  let totalReward = 0;
  let done = false;
  let steps = 0;

  while (!done && steps < maxSteps) {
    const action = policy(state);
    const result = env.step(action);
    transitions.push({
      state,
      action,
      nextState: result.state,
      reward: result.reward,
      done: result.done
    });
    totalReward += result.reward;
    state = result.state;
    done = result.done;
    steps++;
  }

  return {
    transitions,
    totalReward,
    stats: {
      steps,
      totalReward,
      meanReward: steps > 0 ? totalReward / steps : 0,
      length: steps
    }
  };
}
