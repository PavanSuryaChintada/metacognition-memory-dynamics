// Everything on the page comes from the recorded Colab run in ../results.
import metrics from "../../../results/metrics.json";
import replayRun from "../../../results/replay_tracking.json";
import normalisedRun from "../../../results/normalised_replay.json";
import normalisedFigure from "../../../results/figures/normalised.png";
import tenRun from "../../../results/normalised_10_memories.json";
import tenFigure from "../../../results/figures/normalised_10_memories.png";
import forgettingRun from "../../../results/forgetting_5_plus_5.json";
import forgettingFigure from "../../../results/figures/forgetting_5_plus_5.png";
import demoFigure from "../../../results/figures/dynamics_demo.png";
import sweepAFigure from "../../../results/figures/three_regimes_sweep_a.png";
import sweepBFigure from "../../../results/figures/three_regimes_sweep_b.png";
import softmaxFigure from "../../../results/figures/softmax_results.png";
import type { ForgettingTest, NormalisedComparison, Recorded, ReplayTracking, TenMemories } from "./types";

export const recorded = metrics as Recorded;
/** The second Colab run, which logged the softmax replay weights at every step. */
export const replay = replayRun as ReplayTracking;
/** The third Colab run: the same run with raw and with unit-length replay scores. */
export const normalised = normalisedRun as NormalisedComparison;
/** The 10-memory run: the same comparison with six more photos. */
export const ten = tenRun as TenMemories;
/** A test beyond the paper: five memories at step 0, five more added at step 250. */
export const forgetting = forgettingRun as ForgettingTest;
export const figures = {
  demo: demoFigure,
  sweepA: sweepAFigure,
  sweepB: sweepBFigure,
  softmax: softmaxFigure,
  normalised: normalisedFigure,
  ten: tenFigure,
  forgetting: forgettingFigure,
};

/**
 * Replay weights with unit-length scores while every activation is still 1: the score of
 * memory i is the sum of its cosine similarities (including 1 with itself), then
 * softmax(beta * s). Worked out from the measured similarity matrix, not a separate run.
 */
export function normalisedStartShares(beta: number) {
  const s = replay.cosine_similarity.map((row) => row.reduce((a, b) => a + b, 0));
  const m = Math.max(...s);
  const e = s.map((v) => Math.exp(beta * (v - m)));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

/** Display names and thumbnails for every Unsplash photo used in the runs. */
export const PHOTOS: Record<string, { name: string; thumb: string }> = {
  beagle: { name: "Beagle", thumb: "/memories/beagle.jpg" },
  cat: { name: "Cat", thumb: "/memories/cat.jpg" },
  person: { name: "Person", thumb: "/memories/person.jpg" },
  french_bulldog: { name: "French bulldog", thumb: "/memories/french_bulldog.jpg" },
  horse: { name: "Horse", thumb: "/memories/horse.jpg" },
  kitten: { name: "Kitten", thumb: "/memories/kitten.jpg" },
  landscape: { name: "Landscape", thumb: "/memories/landscape.jpg" },
  armchair: { name: "Armchair", thumb: "/memories/armchair.jpg" },
  car: { name: "Car", thumb: "/memories/car.jpg" },
  puppy: { name: "Puppy", thumb: "/memories/puppy.jpg" },
};

export const nameOf = (label: string) => PHOTOS[label]?.name ?? label;

export const memories = recorded.memories.map((m) => ({
  ...m,
  ...PHOTOS[m.label],
  final: recorded.single_run.final_activations[m.label],
}));

export const REPO = "https://github.com/PavanSuryaChintada/metacognition-memory-dynamics";
export const AUTHOR = "Pavan Surya Chintada";
export const AUTHOR_GITHUB = "https://github.com/PavanSuryaChintada";

/**
 * The forgetting test's ten memories. "horse_random" is the random vector the notebook used when the
 * horse photo's link was dead, so it has no thumbnail.
 */
export const forgettingMemories = forgetting.memories.map((m) => ({
  ...m,
  name: m.no_photo ? "Random vector" : PHOTOS[m.label].name,
  thumb: m.no_photo ? undefined : PHOTOS[m.label].thumb,
}));

/** The ten memories of the 10-memory run, with names, thumbnails and printed decay rates. */
export const tenMemories = ten.memories.map((m) => ({
  label: m.label,
  label_in_run: m.label_in_run,
  ...PHOTOS[m.label],
  decay_rate: m.decay_rate_printed,
}));

/** GitHub links to the executed Colab notebooks. */
const nb = (path: string) => `${REPO}/blob/main/notebooks/${path}`;
export const NOTEBOOKS = {
  original: nb("metacognition.ipynb"),
  unnormalised4: nb("colab/unnormalised/metacognition_4_memories.ipynb"),
  normalised4: nb("colab/normalised/normalised_metacognition.ipynb"),
  normalised10: nb("colab/normalised/metacognition_10_mem.ipynb"),
  forgetting: nb("colab/normalised/beyond_paper/5_5_forgetting_test.ipynb"),
  ijepaSample: nb("colab/_facebook_ijepa_vith14_1k.ipynb"),
  vjepaSample: nb("colab/facebook_vjepa2_vitl_fpc64_256.ipynb"),
};
