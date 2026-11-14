// Values worked out from the recorded results, shared by the pages.
import { memories, normalised, recorded, REPO, replay } from "./results";

export const run = recorded.single_run;
export const sweepB = recorded.regimes.sweep_b.runs;
export const decays = recorded.memories.map((m) => m.decay_rate);
export const minDecay = Math.min(...decays);
export const slowest = recorded.memories[decays.indexOf(minDecay)];
export const survivor = memories.find((m) => m.final > 0)!;
/** E(a) with every activation at 1: the Laplacian term vanishes, leaving decay and inhibition. */
export const startEnergy =
  0.5 * decays.reduce((s, d) => s + d, 0) + 0.5 * recorded.base_params.delta * decays.length ** 2;
export const startNorm = Math.sqrt(recorded.memories.length);
export const drive = (r: { eta: number; replay_prob: number }) => +(r.eta * r.replay_prob).toFixed(4);
export const file = (path: string) => `${REPO}/blob/main/${path}`;

/** Final activations of the notebook run, keyed by label. */
export const unnormalisedFinals = recorded.single_run.final_activations;
export const normalisedFinals = normalised.final_activations.normalised;
export const lateShares = normalised.replay_weight_mean_last_10pct.normalised;
const shareValues = Object.values(lateShares);
export const lateShareRange = [Math.min(...shareValues), Math.max(...shareValues)] as const;
export const winnerShare = replay.replay_weight_mean.last_10pct_of_events[survivor.label];
