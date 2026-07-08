# browser-rl-showcase

Browser-based DQN/PPO training showcase with TensorFlow.js — loss curves, live agent visualization, and environment comparison.

> A clean-room re-implementation of the DQN agent that lives in
> [oliverlaudan-ops/AI-Idle](https://github.com/oliverlaudan-ops/AI-Idle)'s
> `src/systems/rl-bot/`, generalized into a standalone, testable, runnable
> showcase.

Live at **<https://rl-showcase.future-pulse.de>** (or locally via `npm run dev`).

## Why this exists

AI-Idle's RL bot is the most interesting part of an otherwise typical idle
game — but it's hidden behind a tab in a 24,000-LOC codebase, and its 27D state
vector is meaningless without understanding the game. This showcase extracts
the algorithm, replaces the AI-Idle-specific state with a generic
4D CartPole vector, and exposes the training loop in the browser so anyone can
watch the loss drop and the agent learn.

## What's in here

```
src/
├── core/                # Reusable RL building blocks (no environment knowledge)
│   ├── replay-buffer.ts    # Experience replay + Prioritized (PER, Schaul et al.)
│   └── reward.ts           # GAE, return, normalization, potential shaping
├── algorithms/          # Agent implementations
│   ├── q-network.ts        # QNetwork interface + MLPQNetwork default
│   └── dqn-agent.ts        # DQN with online+target, ε-greedy, Bellman update
├── envs/                # Toy environments
│   ├── environment.ts      # Environment<S, A> contract + runEpisode helper
│   └── cartpole.ts         # CartPole-v1 (Barto-Sutton-Anderson physics)
└── ui/                  # Browser dashboard
    ├── app.ts              # Training loop, controls, eval episodes
    ├── charts.ts           # Chart.js wrappers (loss + reward)
    ├── cartpole-renderer.ts # 2D canvas visualization
    └── style.css           # Dark theme

tests/                   # Vitest + jsdom + TensorFlow.js CPU backend
```

## Quick start

```bash
npm install
npm run dev          # http://127.0.0.1:5173
```

```bash
npm test             # 117 tests, ~10s
npm run test:coverage
npm run build        # → dist/
```

## Architecture

Three layers, no leakage:

1. **`core/`** — pure utilities. `ReplayBuffer`, `RewardFn`, `calculateAdvantages`. No knowledge of any environment.
2. **`algorithms/`** — agents. `DQNAgent` uses a `QNetwork` (default: `MLPQNetwork`). To swap in a dueling network, conv-net, or anything else, implement the 4-method `QNetwork` interface.
3. **`envs/`** — environments. `Environment<S, A>` is generic. CartPole is the first; adding Gridworld or a trading game is a 100-line change.

The UI layer (`ui/`) wires the agent and environment together with a training loop and live charts.

## What's different from AI-Idle

| Concern | AI-Idle | browser-rl-showcase |
|---|---|---|
| State type | 27D Float32Array (game-specific) | Generic `<S = Float32Array>`, can be any structurally cloneable type |
| Network | Hard-coded MLP in agent | `QNetwork` interface, swappable implementations |
| Reward | 200-line switch-case over game mechanics | `RewardFn<S, A>` callback, environment provides its own |
| Replay | `ReplayBuffer` (no PER implementation) | `ReplayBuffer` + `PrioritizedReplayBuffer` with real PER (Schaul 2016) |
| Memory | `tf-memory-monitor.js` (post-hoc patch) | `tf.tidy` + explicit disposes throughout |
| Tests | None (0% coverage) | 117 tests, 99.45% line coverage |
| Build | None (vanilla JS, ships in `<script>`) | Vite + TypeScript strict mode |

## Deployment

The showcase is deployed at `rl-showcase.future-pulse.de` via the standard
Hostinger DNS + nginx + docker pattern used for other `future-pulse.de`
subdomains (see `crows.future-pulse.de` for the most recent reference
deployment).

Build locally:
```bash
npm run build    # → dist/
```

The output is a static site. The 1.5 MB TF.js chunk is the bulk — gzipped
247 kB. For production, consider:
- Loading TF.js lazily on first training start
- Using the smaller `@tensorflow/tfjs-core` + `@tensorflow/tfjs-converter` +
  `@tensorflow/tfjs-backend-webgl` (instead of the bundled `@tensorflow/tfjs`)
- Service worker caching for repeat visits

## Roadmap

- [ ] **Dueling DQN** — split Q into V(s) + A(s, a). Same `QNetwork` interface.
- [ ] **PPO** — actor-critic, no replay buffer needed. Reuses the `Environment` and `RewardFn` interfaces.
- [ ] **Gridworld toy env** — discrete 5×5 grid, simpler than CartPole, ideal for visualizing the policy as a heatmap.
- [ ] **Policy heatmap** — render the Q-values as a color overlay on the canvas.
- [ ] **TensorBoard-style loss export** — download a `.csv` of training metrics.

## License

MIT
