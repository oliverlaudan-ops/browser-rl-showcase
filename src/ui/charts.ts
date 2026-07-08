/**
 * Chart wrappers for the training dashboard.
 *
 * Two charts: Loss (per training step) and Episode Reward (per completed episode).
 * Each chart keeps a rolling window of the last N points to avoid unbounded growth.
 *
 * Chart.js is used for plotting. We use the bundled version that Vite splits out.
 */

import { Chart, registerables, type ChartConfiguration } from 'chart.js';

Chart.register(...registerables);

const ROLLING_WINDOW = 200;

interface LossPoint {
  step: number;
  value: number;
}

interface RewardPoint {
  episode: number;
  totalReward: number;
  length: number;
  isEval?: boolean;
}

abstract class BaseChart {
  protected chart: Chart;
  protected data: LossPoint[] | RewardPoint[] = [];

  constructor(canvasId: string, config: ChartConfiguration) {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    this.chart = new Chart(canvas, config);
  }

  clear(): void {
    this.data = [];
    this.chart.data.labels = [];
    this.chart.data.datasets.forEach((ds) => {
      ds.data = [];
    });
    this.chart.update('none');
  }
}

export class LossChart extends BaseChart {
  constructor(canvasId: string) {
    super(canvasId, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Loss (MSE)',
            data: [],
            borderColor: '#ff5d5d',
            backgroundColor: 'rgba(255, 93, 93, 0.1)',
            borderWidth: 1,
            pointRadius: 0,
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false }
        },
        scales: {
          x: { display: false },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9aa3b2', maxTicksLimit: 4 }
          }
        }
      }
    });
  }

  push(loss: number): void {
    const point: LossPoint = { step: (this.data as LossPoint[]).length, value: loss };
    (this.data as LossPoint[]).push(point);
    if ((this.data as LossPoint[]).length > ROLLING_WINDOW) {
      (this.data as LossPoint[]).shift();
    }
    this.chart.data.labels = (this.data as LossPoint[]).map((p) => p.step);
    this.chart.data.datasets[0]!.data = (this.data as LossPoint[]).map((p) => p.value);
    this.chart.update('none');
  }
}

export class RewardChart extends BaseChart {
  constructor(canvasId: string) {
    super(canvasId, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Train reward',
            data: [],
            borderColor: '#3ddc97',
            backgroundColor: 'rgba(61, 220, 151, 0.1)',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.2
          },
          {
            label: 'Eval reward',
            data: [],
            borderColor: '#4c8eff',
            backgroundColor: 'rgba(76, 142, 255, 0.1)',
            borderWidth: 1.5,
            pointRadius: 3,
            pointBackgroundColor: '#4c8eff',
            showLine: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            display: true,
            labels: { color: '#9aa3b2', boxWidth: 12, font: { size: 11 } }
          },
          tooltip: { enabled: false }
        },
        scales: {
          x: { display: false },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9aa3b2', maxTicksLimit: 4 }
          }
        }
      }
    });
  }

  push(totalReward: number, length: number): void {
    const point: RewardPoint = {
      episode: (this.data as RewardPoint[]).length,
      totalReward,
      length
    };
    (this.data as RewardPoint[]).push(point);
    if ((this.data as RewardPoint[]).length > ROLLING_WINDOW) {
      (this.data as RewardPoint[]).shift();
    }
    this.refresh();
  }

  pushEval(totalReward: number, length: number): void {
    const lastEpisode =
      ((this.data as RewardPoint[]).at(-1)?.episode ?? 0) + 1;
    const point: RewardPoint = {
      episode: lastEpisode,
      totalReward,
      length,
      isEval: true
    };
    (this.data as RewardPoint[]).push(point);
    this.refresh();
  }

  private refresh(): void {
    this.chart.data.labels = (this.data as RewardPoint[]).map((p) => p.episode);
    this.chart.data.datasets[0]!.data = (this.data as RewardPoint[])
      .filter((p) => !p.isEval)
      .map((p) => p.totalReward);
    this.chart.data.datasets[1]!.data = (this.data as RewardPoint[])
      .filter((p) => p.isEval)
      .map((p) => ({ x: p.episode, y: p.totalReward }));
    this.chart.update('none');
  }
}
