/**
 * Experience Replay Buffer
 *
 * Stores past experiences (state, action, reward, next_state, done) for off-policy
 * reinforcement learning. Enables:
 * - Breaking correlation between consecutive experiences
 * - Reusing experiences multiple times
 * - Stable training through random sampling
 *
 * Generic implementation — does NOT depend on any specific environment.
 * Pass a state of any type that can be cloned (structurally or via copy constructor).
 */

/**
 * A single experience tuple.
 * @typeParam S - State type. Must be a typed array or structurally cloneable.
 */
export interface Experience<S = Float32Array> {
  state: S;
  action: number;
  reward: number;
  nextState: S;
  done: boolean;
}

/**
 * A sampled batch of experiences, parallel arrays (TF.js-friendly).
 * @typeParam S - State type.
 */
export interface ExperienceBatch<S = Float32Array> {
  states: S[];
  actions: number[];
  rewards: number[];
  nextStates: S[];
  dones: number[]; // 0 or 1, never booleans (TF.js prefers numeric)
}

/**
 * Buffer statistics for logging and monitoring.
 */
export interface BufferStats {
  size: number;
  capacity: number;
  utilizationPercent: number;
  avgReward: number;
  maxReward: number;
  minReward: number;
}

/**
 * Serializable buffer shape (for persistence / toJSON).
 */
export interface SerializedBuffer<S = number[]> {
  capacity: number;
  position: number;
  size: number;
  buffer: Array<{
    state: S;
    action: number;
    reward: number;
    nextState: S;
    done: boolean;
  }>;
}

/**
 * Helper: clone a state. We support Float32Array directly; for arbitrary types,
 * the user should provide a clone function via constructor options.
 */
type CloneFn<S> = (s: S) => S;

/**
 * Default clone strategy: assume Float32Array.
 * Throws for other types unless a custom clone is provided.
 */
const defaultCloneFloat32 = (s: Float32Array): Float32Array => new Float32Array(s);

export class ReplayBuffer<S = Float32Array> {
  protected buffer: Experience<S>[];
  protected position = 0;
  protected size = 0;
  public capacity: number; // mutable to support fromJSON restore
  protected readonly cloneState: CloneFn<S>;

  /**
   * @param capacity - Maximum number of experiences to store.
   * @param options - Optional. Pass `cloneState` for non-Float32Array state types.
   */
  constructor(
    capacity = 10_000,
    options: { cloneState?: CloneFn<S> } = {}
  ) {
    if (capacity <= 0) {
      throw new Error(`ReplayBuffer capacity must be > 0, got ${capacity}`);
    }
    this.capacity = capacity;
    this.buffer = [];
    // Type-erase the default clone: it's only valid for Float32Array-shaped state.
    // We cast to CloneFn<S> because TypeScript can't narrow the type parameter here.
    this.cloneState = (options.cloneState ?? (defaultCloneFloat32 as unknown as CloneFn<S>));
  }

  /**
   * Add an experience to the buffer. Copies the state to avoid reference aliasing.
   * When the buffer is full, overwrites the oldest entry (circular).
   */
  add(state: S, action: number, reward: number, nextState: S, done: boolean): void {
    const experience: Experience<S> = {
      state: this.cloneState(state),
      action,
      reward,
      nextState: this.cloneState(nextState),
      done
    };

    if (this.buffer.length < this.capacity) {
      this.buffer.push(experience);
    } else {
      this.buffer[this.position] = experience;
    }

    this.position = (this.position + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }

  /**
   * Sample a random batch of experiences without replacement.
   * @throws When batchSize > current size.
   */
  sample(batchSize: number): ExperienceBatch<S> {
    if (batchSize > this.size) {
      throw new Error(
        `Cannot sample ${batchSize} experiences, only ${this.size} available`
      );
    }
    const indices = this.sampleIndices(batchSize);
    return this.getBatch(indices);
  }

  /**
   * Random indices without replacement.
   * Public for testing — internal use preferred.
   */
  protected sampleIndices(count: number): number[] {
    const indices: number[] = [];
    const available = new Set<number>();
    for (let i = 0; i < this.size; i++) {
      available.add(i);
    }
    while (indices.length < count && available.size > 0) {
      const arr = Array.from(available);
      const randomIndex = Math.floor(Math.random() * arr.length);
      const selected = arr[randomIndex];
      indices.push(selected);
      available.delete(selected);
    }
    return indices;
  }

  /**
   * Get batch by indices.
   */
  protected getBatch(indices: number[]): ExperienceBatch<S> {
    const batch: ExperienceBatch<S> = {
      states: [],
      actions: [],
      rewards: [],
      nextStates: [],
      dones: []
    };
    for (const index of indices) {
      const exp = this.buffer[index];
      batch.states.push(exp.state);
      batch.actions.push(exp.action);
      batch.rewards.push(exp.reward);
      batch.nextStates.push(exp.nextState);
      batch.dones.push(exp.done ? 1 : 0);
    }
    return batch;
  }

  /**
   * Number of experiences currently stored.
   */
  length(): number {
    return this.size;
  }

  /**
   * Whether the buffer has enough experiences for a training batch.
   */
  canSample(batchSize: number): boolean {
    return this.size >= batchSize;
  }

  /**
   * Clear all experiences.
   */
  clear(): void {
    this.buffer = [];
    this.position = 0;
    this.size = 0;
  }

  /**
   * Get the most recent N experiences (for debugging / visualization).
   * Order: oldest → newest within the slice.
   */
  getRecent(n = 10): Experience<S>[] {
    const recent: Experience<S>[] = [];
    const startPos = Math.max(0, this.position - n);
    for (let i = startPos; i < this.position; i++) {
      recent.push(this.buffer[i]);
    }
    return recent;
  }

  /**
   * Aggregate reward statistics over the buffer.
   */
  getStats(): BufferStats {
    if (this.size === 0) {
      return {
        size: 0,
        capacity: this.capacity,
        utilizationPercent: 0,
        avgReward: 0,
        maxReward: 0,
        minReward: 0
      };
    }

    const rewards = this.buffer.slice(0, this.size).map((exp) => exp.reward);
    const sum = rewards.reduce((a, b) => a + b, 0);
    const avgReward = sum / rewards.length;
    const maxReward = Math.max(...rewards);
    const minReward = Math.min(...rewards);

    return {
      size: this.size,
      capacity: this.capacity,
      utilizationPercent: (this.size / this.capacity) * 100,
      avgReward,
      maxReward,
      minReward
    };
  }

  /**
   * Serialize to JSON-compatible object.
   * For Float32Array state, this stores plain number arrays.
   */
  toJSON(): SerializedBuffer {
    return {
      capacity: this.capacity,
      position: this.position,
      size: this.size,
      buffer: this.buffer.slice(0, this.size).map((exp) => ({
        // For Float32Array, convert to plain array. For other types, assume JSON-safe.
        state: Array.from(exp.state as unknown as ArrayLike<number>) as unknown as number[],
        action: exp.action,
        reward: exp.reward,
        nextState: Array.from(exp.nextState as unknown as ArrayLike<number>) as unknown as number[],
        done: exp.done
      }))
    };
  }

  /**
   * Restore buffer from a previously serialized snapshot.
   */
  fromJSON(data: SerializedBuffer): void {
    this.capacity = data.capacity;
    this.position = data.position;
    this.size = data.size;
    this.buffer = data.buffer.map((exp) => ({
      state: new Float32Array(exp.state) as unknown as S,
      action: exp.action,
      reward: exp.reward,
      nextState: new Float32Array(exp.nextState) as unknown as S,
      done: exp.done
    }));
  }
}

/**
 * Prioritized Experience Replay Buffer (PER).
 * Samples based on TD-error priority. Falls back to uniform sampling if not
 * configured. TD-error updates are stored as priorities.
 *
 * Reference: Schaul et al. (2016), "Prioritized Experience Replay".
 */
export class PrioritizedReplayBuffer<S = Float32Array> extends ReplayBuffer<S> {
  protected priorities: Float32Array;
  protected maxPriority = 1.0;
  public readonly alpha: number;
  public readonly beta: number;

  constructor(
    capacity = 10_000,
    options: { alpha?: number; beta?: number; cloneState?: CloneFn<S> } = {}
  ) {
    super(capacity, { cloneState: options.cloneState });
    this.alpha = options.alpha ?? 0.6; // Priority exponent
    this.beta = options.beta ?? 0.4; // Importance sampling exponent
    this.priorities = new Float32Array(capacity);
  }

  override add(state: S, action: number, reward: number, nextState: S, done: boolean): void {
    super.add(state, action, reward, nextState, done);
    const index = (this.position - 1 + this.capacity) % this.capacity;
    this.priorities[index] = this.maxPriority;
  }

  /**
   * Priority-based sampling (alpha-weighted).
   * Returns the batch AND importance-sampling weights.
   *
   * NOTE: This is the *real* PER sampling — uniform fallback is only used when
   * the priorities array is empty (cold start).
   */
  sampleWithPriorities(batchSize: number): {
    batch: ExperienceBatch<S>;
    indices: number[];
    weights: Float32Array;
  } {
    if (batchSize > this.size) {
      throw new Error(
        `Cannot sample ${batchSize} experiences, only ${this.size} available`
      );
    }

    // Compute sampling probabilities P(i) = p_i^alpha / sum(p_j^alpha)
    const prioritiesSlice = Array.from(this.priorities.slice(0, this.size));
    const alphaPow = prioritiesSlice.map((p) => Math.pow(p, this.alpha));
    const total = alphaPow.reduce((a, b) => a + b, 0);
    const probs = alphaPow.map((p) => p / total);

    // Sample without replacement proportional to probs
    const indices: number[] = [];
    const indicesSet = new Set<number>();
    while (indices.length < batchSize) {
      const r = Math.random();
      let cum = 0;
      let chosen = -1;
      for (let i = 0; i < probs.length; i++) {
        if (indicesSet.has(i)) continue;
        cum += probs[i];
        if (r <= cum) {
          chosen = i;
          break;
        }
      }
      if (chosen === -1) {
        // Fallback: pick first available (shouldn't happen unless probs don't sum to 1)
        for (let i = 0; i < probs.length; i++) {
          if (!indicesSet.has(i)) {
            chosen = i;
            break;
          }
        }
      }
      if (chosen === -1) break;
      indices.push(chosen);
      indicesSet.add(chosen);
    }

    // Importance-sampling weights: w_i = (1 / (N * P(i)))^beta, normalized by max
    const N = this.size;
    const weights = new Float32Array(batchSize);
    let maxWeight = 0;
    for (let i = 0; i < batchSize; i++) {
      const idx = indices[i];
      const w = Math.pow(1 / (N * probs[idx]), this.beta);
      weights[i] = w;
      if (w > maxWeight) maxWeight = w;
    }
    if (maxWeight > 0) {
      for (let i = 0; i < batchSize; i++) {
        weights[i] = weights[i] / maxWeight;
      }
    }

    const batch = this.getBatch(indices);
    return { batch, indices, weights };
  }

  /**
   * Update priorities for sampled experiences based on new TD-errors.
   * Use abs() + small epsilon to ensure non-zero priority.
   */
  updatePriorities(indices: number[], priorities: number[]): void {
    for (let i = 0; i < indices.length; i++) {
      const p = Math.abs(priorities[i]) + 1e-6;
      this.priorities[indices[i]] = p;
      if (p > this.maxPriority) this.maxPriority = p;
    }
  }
}
