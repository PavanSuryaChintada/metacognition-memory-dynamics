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
