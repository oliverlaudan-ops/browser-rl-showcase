/**
 * CartPole Environment
 *
 * Classic control task: balance a pole on a cart by moving left or right.
 * The agent observes a 4D state (cart position, cart velocity, pole angle, pole angular velocity)
 * and outputs a discrete action (0 = push left, 1 = push right).
 *
 * Reward: +1 for every timestep the pole stays upright.
 * Episode ends when: pole angle > 12° OR cart position > 2.4 OR step limit (default 500) reached.
 *
 * Physics based on Barto, Sutton, Anderson (1983), with Euler integration.
 * This is a *pure TypeScript* implementation — no canvas, no physics engine.
 * Browser-side rendering of the cart/pole is the UI's job.
 *
 * Provenance: this is the most-cited RL benchmark. We include it because:
 * - It trains in <60s on a modern browser with DQN.
 * - It's a known "shape" — if your algorithm doesn't solve this, the bug is in the algorithm.
 */

import type { Environment, EnvSpec, StepResult } from './environment.js';

const GRAVITY = 9.8;
const CART_MASS = 1.0;
const POLE_MASS = 0.1;
const TOTAL_MASS = CART_MASS + POLE_MASS;
const POLE_LENGTH = 0.5; // half-length, actually (distance from pivot to COM)
// POLE_MOMENT unused (kept for future extension of physics)
const FORCE_MAG = 10.0;
const TAU = 0.02; // seconds per step

const DEFAULT_MAX_STEPS = 500;
const DEFAULT_X_THRESHOLD = 2.4;
const DEFAULT_THETA_THRESHOLD_RAD = (12 * Math.PI) / 180; // 12 degrees

export interface CartPoleConfig {
  maxSteps?: number;
  xThreshold?: number;
  thetaThresholdRadians?: number;
  initialStateRange?: number;
}

export class CartPoleEnv implements Environment<Float32Array, number> {
  private state: Float32Array = new Float32Array(4);
  private steps = 0;
  private readonly maxSteps: number;
  private readonly xThreshold: number;
  private readonly thetaThreshold: number;
  private readonly initialStateRange: number;
  private readonly id: string;

  constructor(config: CartPoleConfig = {}, id = 'cartpole-v1') {
    this.maxSteps = config.maxSteps ?? DEFAULT_MAX_STEPS;
    this.xThreshold = config.xThreshold ?? DEFAULT_X_THRESHOLD;
    this.thetaThreshold = config.thetaThresholdRadians ?? DEFAULT_THETA_THRESHOLD_RAD;
    this.initialStateRange = config.initialStateRange ?? 0.05;
    this.id = id;
  }

  reset(): Float32Array {
    // Initialize with small uniform random offsets
    this.state[0] = (Math.random() * 2 - 1) * this.initialStateRange; // x
    this.state[1] = (Math.random() * 2 - 1) * this.initialStateRange; // x_dot
    this.state[2] = (Math.random() * 2 - 1) * this.initialStateRange; // theta
    this.state[3] = (Math.random() * 2 - 1) * this.initialStateRange; // theta_dot
    this.steps = 0;
    return new Float32Array(this.state);
  }

  step(action: number): StepResult<Float32Array> {
    if (action !== 0 && action !== 1) {
      throw new Error(`Invalid action ${action}. CartPole is discrete: 0 or 1.`);
    }

    const force = action === 1 ? FORCE_MAG : -FORCE_MAG;
    const [x, xDot, theta, thetaDot] = this.state;

    // Standard CartPole physics
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);

    const temp =
      (force + POLE_MASS * POLE_LENGTH * thetaDot * thetaDot * sinTheta) / TOTAL_MASS;
    const thetaAcc =
      (GRAVITY * sinTheta - cosTheta * temp) /
      (POLE_LENGTH * (4.0 / 3.0 - (POLE_MASS * cosTheta * cosTheta) / TOTAL_MASS));
    const xAcc = temp - (POLE_MASS * POLE_LENGTH * thetaAcc * cosTheta) / TOTAL_MASS;

    // Euler integration
    const newX = x + TAU * xDot;
    const newXDot = xDot + TAU * xAcc;
    const newTheta = theta + TAU * thetaDot;
    const newThetaDot = thetaDot + TAU * thetaAcc;

    this.state[0] = newX;
    this.state[1] = newXDot;
    this.state[2] = newTheta;
    this.state[3] = newThetaDot;
    this.steps++;

    // Termination checks
    const outOfBounds =
      newX < -this.xThreshold ||
      newX > this.xThreshold ||
      newTheta < -this.thetaThreshold ||
      newTheta > this.thetaThreshold;
    const truncated = this.steps >= this.maxSteps;
    const done = outOfBounds || truncated;

    return {
      state: new Float32Array(this.state),
      reward: 1.0,
      done,
      info: {
        steps: this.steps,
        x: newX,
        theta: newTheta,
        termination: outOfBounds ? 'out_of_bounds' : truncated ? 'truncated' : null
      }
    };
  }

  getSpec(): EnvSpec {
    return {
      id: this.id,
      observationDim: 4,
      actionDim: 2,
      actionType: 'discrete',
      maxSteps: this.maxSteps,
      rewardRange: [0, 1] as const
    };
  }
}
