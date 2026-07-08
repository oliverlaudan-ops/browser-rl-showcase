/**
 * App entry point — wires the UI to the DQN agent and CartPole environment.
 *
 * Training loop: in async microtask + setTimeout, runs at user-selected speed.
 * Between steps, the main thread can paint — at 50 steps/s with rAF this is smooth.
 *
 * Why not a Web Worker? The TF.js model lives in the main thread. Moving it
 * to a worker would require copying the Float32Array state back and forth on
 * every selectAction call, which is slower than just yielding the main thread
 * with setTimeout(0). For a showcase, the simpler path wins.
 */

import { DQNAgent, type DQNAgentConfig } from '../algorithms/dqn-agent.js';
import { CartPoleEnv } from '../envs/cartpole.js';
import { LossChart, RewardChart } from './charts.js';
import { CartPoleRenderer } from './cartpole-renderer.js';

interface HyperparamPreset {
  name: string;
  config: Partial<DQNAgentConfig>;
}

const PRESETS: Record<string, HyperparamPreset> = {
  default: {
    name: 'Default',
    config: {
      epsilonStart: 1.0,
      epsilonEnd: 0.05,
      epsilonDecay: 0.995,
      learningRate: 0.001,
      targetUpdateFreq: 100
    }
  },
  fast: {
    name: 'Fast',
    config: {
      epsilonStart: 0.5,
      epsilonEnd: 0.01,
      epsilonDecay: 0.99,
      learningRate: 0.002,
      targetUpdateFreq: 50
    }
  },
  explore: {
    name: 'Explore',
    config: {
      epsilonStart: 1.0,
      epsilonEnd: 0.2,
      epsilonDecay: 0.999,
      learningRate: 0.0005,
      targetUpdateFreq: 200
    }
  }
};

class App {
  private agent: DQNAgent | null = null;
  private env: CartPoleEnv | null = null;
  private lossChart: LossChart;
  private rewardChart: RewardChart;
  private renderer: CartPoleRenderer;

  private running = false;
  private speed = 1;
  private currentEpisode = 0;
  private currentStep = 0;
  private currentState: Float32Array | null = null;
  private currentEpisodeReward = 0;
  private currentEpisodeLength = 0;

  constructor() {
    this.lossChart = new LossChart('lossChart');
    this.rewardChart = new RewardChart('rewardChart');
    this.renderer = new CartPoleRenderer('cartpoleCanvas');

    this.bindControls();
    this.refreshStats();
  }

  private bindControls(): void {
    const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
    const stopBtn = document.getElementById('stopBtn') as HTMLButtonElement;
    const evalBtn = document.getElementById('evalBtn') as HTMLButtonElement;
    const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
    const speedSlider = document.getElementById('speedSlider') as HTMLInputElement;
    const speedLabel = document.getElementById('speedLabel') as HTMLSpanElement;
    const presetSelect = document.getElementById('hyperparamPreset') as HTMLSelectElement;

    startBtn.addEventListener('click', () => this.start());
    stopBtn.addEventListener('click', () => this.stop());
    evalBtn.addEventListener('click', () => this.runEvalEpisode());
    resetBtn.addEventListener('click', () => this.reset());

    speedSlider.addEventListener('input', () => {
      this.speed = parseInt(speedSlider.value, 10);
      speedLabel.textContent = `${this.speed}×`;
    });

    // Disable preset switching during training
    presetSelect.addEventListener('change', () => {
      if (this.running) {
        presetSelect.value = this.currentPresetKey;
      }
    });
  }

  private currentPresetKey = 'default';
  private selectedPresetKey(): string {
    const sel = document.getElementById('hyperparamPreset') as HTMLSelectElement;
    return sel.value;
  }

  private buildAgentConfig(): DQNAgentConfig {
    const preset = PRESETS[this.selectedPresetKey()] ?? PRESETS.default!;
    this.currentPresetKey = this.selectedPresetKey();
    return {
      actionDim: 2,
      stateDim: 4,
      bufferCapacity: 10_000,
      batchSize: 32,
      ...preset.config
    };
  }

  private start(): void {
    if (this.running) return;
    this.running = true;
    this.setControlsState(true);
    this.spawnAgent();
    this.env = new CartPoleEnv({ maxSteps: 500 });
    this.currentState = this.env.reset();
    this.currentStep = 0;
    this.currentEpisodeReward = 0;
    this.currentEpisodeLength = 0;
    this.runLoop();
  }

  private stop(): void {
    this.running = false;
    this.setControlsState(false);
  }

  private reset(): void {
    this.stop();
    if (this.agent) {
      this.agent.dispose();
      this.agent = null;
    }
    this.env = null;
    this.currentState = null;
    this.currentEpisode = 0;
    this.currentStep = 0;
    this.currentEpisodeReward = 0;
    this.currentEpisodeLength = 0;
    this.lossChart.clear();
    this.rewardChart.clear();
    this.renderer.clear();
    this.refreshStats();
  }

  private spawnAgent(): void {
    this.agent = new DQNAgent(this.buildAgentConfig());
  }

  private async runLoop(): Promise<void> {
    while (this.running && this.agent && this.env && this.currentState) {
      // One environment step
      const state = this.currentState;
      const action = await this.agent.selectAction(state, true);
      const result = this.env.step(action);

      this.agent.remember(state, action, result.reward, result.state, result.done);
      const loss = await this.agent.trainStep();
      if (Number.isFinite(loss)) {
        this.lossChart.push(loss);
      }

      this.currentState = result.state;
      this.currentStep++;
      this.currentEpisodeReward += result.reward;
      this.currentEpisodeLength = this.currentStep;

      // Render at the end of the speed-batched steps
      this.renderer.render(result.state, action);
      this.refreshStats();

      if (result.done) {
        this.currentEpisode++;
        this.rewardChart.push(this.currentEpisodeReward, this.currentEpisodeLength);
        this.currentState = this.env.reset();
        this.currentStep = 0;
        this.currentEpisodeReward = 0;
        this.currentEpisodeLength = 0;
      }

      // Yield to the browser between batches of `speed` steps
      if (this.currentStep % Math.max(1, this.speed) === 0) {
        await new Promise((r) => setTimeout(r, 0));
      }
    }
  }

  private async runEvalEpisode(): Promise<void> {
    if (!this.agent) {
      this.spawnAgent();
    }
    if (!this.agent) return;
    const env = new CartPoleEnv({ maxSteps: 500 });
    let state = env.reset();
    let totalReward = 0;
    let steps = 0;
    let done = false;
    this.renderer.render(state, 0);
    while (!done && steps < 500) {
      const action = await this.agent.selectAction(state, false);
      const result = env.step(action);
      totalReward += result.reward;
      steps++;
      this.renderer.render(result.state, action);
      // Yield occasionally to keep the UI responsive
      if (steps % 10 === 0) {
        await new Promise((r) => setTimeout(r, 0));
      }
      state = result.state;
      done = result.done;
    }
    this.rewardChart.pushEval(totalReward, steps);
    this.refreshStats();
  }

  private setControlsState(running: boolean): void {
    (document.getElementById('startBtn') as HTMLButtonElement).disabled = running;
    (document.getElementById('stopBtn') as HTMLButtonElement).disabled = !running;
    (document.getElementById('evalBtn') as HTMLButtonElement).disabled = running;
    (document.getElementById('resetBtn') as HTMLButtonElement).disabled = false;
    (document.getElementById('hyperparamPreset') as HTMLSelectElement).disabled = running;
  }

  private refreshStats(): void {
    const stats = this.agent?.getStats();
    (document.getElementById('episodeStat') as HTMLElement).textContent = String(
      this.currentEpisode
    );
    (document.getElementById('stepStat') as HTMLElement).textContent = String(this.currentStep);
    (document.getElementById('lastRewardStat') as HTMLElement).textContent = this.currentEpisodeReward.toFixed(0);
    (document.getElementById('episodeLengthStat') as HTMLElement).textContent = String(this.currentEpisodeLength);
    if (stats) {
      (document.getElementById('bufferStat') as HTMLElement).textContent =
        `${stats.bufferSize} / 10000`;
      (document.getElementById('epsilonStat') as HTMLElement).textContent =
        stats.epsilon.toFixed(3);
      (document.getElementById('lossStat') as HTMLElement).textContent = Number.isFinite(stats.loss)
        ? stats.loss.toFixed(4)
        : '—';
      (document.getElementById('meanQStat') as HTMLElement).textContent = Number.isFinite(stats.meanQ)
        ? stats.meanQ.toFixed(2)
        : '—';
    }
  }
}

// Bootstrap on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new App());
} else {
  new App();
}
