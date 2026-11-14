// Everything on the page comes from the recorded Colab run in ../results.
import metrics from "../../../results/metrics.json";
import replayRun from "../../../results/replay_tracking.json";
import normalisedRun from "../../../results/normalised_replay.json";
import normalisedFigure from "../../../results/figures/normalised.png";
import demoFigure from "../../../results/figures/dynamics_demo.png";
import sweepAFigure from "../../../results/figures/three_regimes_sweep_a.png";
import sweepBFigure from "../../../results/figures/three_regimes_sweep_b.png";
import softmaxFigure from "../../../results/figures/softmax_results.png";
import type { NormalisedComparison, Recorded, ReplayTracking } from "./types";

export const recorded = metrics as Recorded;
/** The second Colab run, which logged the softmax replay weights at every step. */
export const replay = replayRun as ReplayTracking;
/** The third Colab run: the same run with raw and with unit-length replay scores. */
export const normalised = normalisedRun as NormalisedComparison;
export const figures = {
  demo: demoFigure,
  sweepA: sweepAFigure,
  sweepB: sweepBFigure,
  softmax: softmaxFigure,
  normalised: normalisedFigure,
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

/** Display names and thumbnails for the four Unsplash photos, in memory order. */
export const PHOTOS: Record<string, { name: string; thumb: string }> = {
  beagle: { name: "Beagle", thumb: "/memories/beagle.jpg" },
  cat: { name: "Cat", thumb: "/memories/cat.jpg" },
  person: { name: "Person", thumb: "/memories/person.jpg" },
  french_bulldog: { name: "French bulldog", thumb: "/memories/french_bulldog.jpg" },
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
