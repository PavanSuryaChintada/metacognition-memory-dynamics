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
