type Forces = { decay: number; diffusion: number; inhibition: number; replay: number; net: number };

/** Shape of results/replay_tracking.json: the second Colab run, with replay weights logged. */
export type ReplayTracking = {
  setup: { seed: number; steps: number; eta: number; beta: number; replay_prob: number };
  labels: string[];
  labels_in_run: string[];
  replay_events: number;
  cosine_similarity: number[][];
  final_activations: Record<string, number>;
  replay_weight_mean: {
    first_10pct_of_events: Record<string, number>;
    last_10pct_of_events: Record<string, number>;
  };
  mean_forces: Record<string, Forces>;
};

type Arms<T> = { unnormalised: T; normalised: T };

/** Shape of results/normalised_replay.json: the third Colab run, raw vs unit-length replay scores. */
export type NormalisedComparison = {
  setup: {
    steps: number;
    seed: number;
    eta: number;
    beta: number;
    replay_prob: number;
    gamma: number;
    delta: number;
    h: number;
  };
  labels: string[];
  labels_in_run: string[];
  decay_rates_printed: number[];
  replay_events: Arms<number>;
  final_activations: Arms<Record<string, number>>;
  replay_weight_mean_last_10pct: Arms<Record<string, number>>;
  survivors_above_0_001: Arms<number>;
};

/** Shape of results/normalised_10_memories.json: the 10-memory raw vs normalised run. */
export type TenMemories = {
  setup: { steps: number; seed: number; eta: number; beta: number; replay_prob: number };
  memories: { label: string; label_in_run: string; decay_rate_printed: number; url: string }[];
  final_activations: { unnormalised: Record<string, number>; normalised: Record<string, number> };
  survivors_above_0_001: { unnormalised: number; normalised: number };
  order_by_decay_as_printed: string[];
};

/** Shape of results/forgetting_5_plus_5.json: five memories at step 0, five more added at step 250. */
export type ForgettingTest = {
  setup: {
    replay: string;
    steps: number;
    inject_at: number;
    new_memory_start_activation: number;
    seed: number;
    eta: number;
    beta: number;
    replay_prob: number;
    gamma: number;
    delta: number;
    h: number;
    alive_threshold: number;
  };
  memories: {
    label: string;
    label_in_run: string;
    group: "old" | "new";
    born: number;
    decay_rate_printed: number;
    at_250: number;
    final: number;
    status: string;
    no_photo?: boolean;
  }[];
  survivors: { old: number; old_total: number; new: number; new_total: number };
  old_change_250_to_500_pct_printed: Record<string, number>;
  verdict_printed: string;
  old_total_printed: { at_250: number; final: number; change_pct: number };
};

type Arm3<T> = { raw: T; mean: T; warmup: T };

/** Shape of results/forgetting_fixes.json: the forgetting test run three ways. */
export type ForgettingFixes = {
  setup: {
    steps: number;
    inject_at: number;
    seed: number;
    eta: number;
    beta: number;
    replay_prob: number;
    gamma: number;
    delta: number;
    h: number;
    alive_threshold: number;
    warmup_steps: number;
  };
  arms: { id: "raw" | "mean" | "warmup"; name: string; printed: string; how: string }[];
  memories: { label: string; label_in_run: string; group: "old" | "new"; decay_rate_printed: number; final: Arm3<number> }[];
  survivors: Arm3<{ old: number; new: number }>;
  retention_pct_printed: Arm3<number>;
  summary_printed: { max_old: Arm3<number>; max_new: Arm3<number>; old_new_ratio: Arm3<number> };
  best_printed: string;
  old_sum_at_250_derived: number;
};

/** Shape of results/paper_scale.json: run C, the paper's N and T with 200 CIFAR-100 images. */
export type PaperScale = {
  setup: {
    images: string;
    n: number;
    steps: number;
    h: number;
    beta: number;
    gamma: number;
    delta: number;
    eta: number;
    replay_prob: number;
    seed: number;
    alive_threshold: number;
  };
  embedding_norms: { min: number; max: number; mean: number };
  cosine_off_diagonal: { min: number; max: number; mean: number; std: number };
  progress: { step: number; alive: number; max: string; mean: string }[];
  survivors: number;
  max_activation: number;
  mean_activation: number;
  spearman_rho_decay_vs_activation: number;
  spearman_p: string;
  top10: { rank: number; label: string; decay: number; activation: number }[];
  verdict_printed: string;
};

/** Shape of results/paper_tuned.json: run C, test 2, a 5 x 5 sweep of gamma and delta. */
export type PaperTuned = {
  setup: {
    n: number;
    steps: number;
    h: number;
    beta: number;
    eta: number;
    replay_prob: number;
    seed: number;
    alive_threshold: number;
    gammas: number[];
    deltas: number[];
    paper: { gamma: number; delta: number };
    mean_cosine: number;
  };
  grid: { gamma: number; delta: number; alive: number; rho: number; max: number }[];
  best_printed: { gamma: number; delta: number; alive: number; rho: number; max: number; min_alive: number; mean_alive: number };
  selection_rule: string;
  top10_best: { rank: number; label: string; decay: number; activation: number }[];
  survival_by_decay_bin_best: { from: number; to: number; survived: number; of: number }[];
};

/** Shape of results/metrics.json: the numbers the Colab run printed. */
export type Recorded = {
  source: string;
  model: string;
  seed: number;
  steps: number;
  base_params: { gamma: number; delta: number; beta: number; h: number; initial_activation: number };
  memories: { index: number; label: string; label_in_run: string; decay_rate: number; url: string }[];
  single_run: {
    eta: number;
    replay_prob: number;
    final_activations: Record<string, number>;
    final_energy: number;
    retrieval: { query: string; top: { label: string; score: number }[] };
  };
  regimes: {
    sweep_a: { runs: { name: string; eta: number; replay_prob: number; approx_final_norm: number }[] };
    sweep_b: { runs: { name: string; eta: number; replay_prob: number; final_norm: number; final_energy: number }[] };
  };
};
