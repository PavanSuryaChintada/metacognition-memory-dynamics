import type { CSSProperties, ReactNode } from "react";
import {
  bayes200,
  bayesFour,
  bayesRelative,
  fixes,
  fixMemories,
  forgetting,
  forgettingMemories,
  normalised,
  paperScale,
  paperTuned,
  ten,
} from "../data/results";
import { lateShareRange, normalisedFinals, unnormalisedFinals, winnerShare } from "../data/derived";
import { useIsNarrow, usePinnedProgress } from "../hooks";
import { MiniBars } from "./Memories";
import { SurvivorGrid } from "./SurvivorGrid";
import { ThresholdMeter } from "./RunCards";
import { RetentionBars, Timeline } from "../pages/BeyondPaper";
import { CONFIDENCE_RANGES, ConfidenceRanges, neg } from "../pages/RunBayes";
import { Card, Chapter, Section } from "./Layout";

/* ------------------------------------------------------------ numbers, all from the recorded runs */

const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;
const word = (n: number) => ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n] ?? String(n);

const ps = paperScale.setup;
const meanDecay = (0.5 + 1.5) / 2; // run C's decay rates are evenly spaced from 0.5 to 1.5
const deltaN = ps.delta * ps.n;
const holdTotal = (ps.eta * ps.replay_prob) / (meanDecay + deltaN);
const best = paperTuned.best_printed;
const lowDelta = Math.min(...paperTuned.setup.deltas);

const fg = forgetting.setup;
const oldMems = forgettingMemories.filter((m) => m.group === "old");
const oldPhotos = oldMems.filter((m) => !m.no_photo);
const drops = oldPhotos.map((m) => -forgetting.old_change_250_to_500_pct_printed[m.label]);
const sumBefore = oldMems.reduce((s, m) => s + m.at_250, 0);
const sumAfter = sumBefore + (forgettingMemories.length - oldMems.length) * fg.new_memory_start_activation;

const ret = fixes.retention_pct_printed;
const fixOldDecays = fixMemories.filter((m) => m.group === "old").map((m) => m.decay_rate_printed);
const decayOnly = (alpha: number) => (1 - fixes.setup.h * alpha) ** (fixes.setup.steps - fixes.setup.inject_at);
const decayRange = [decayOnly(Math.max(...fixOldDecays)), decayOnly(Math.min(...fixOldDecays))];

const same = (a: { label: string; decay: number }, b: { label: string; decay: number }) => a.label === b.label && a.decay === b.decay;
const topMatch = bayesRelative.most_confident_top5.filter((m) => bayes200.suppressed_top5.some((s) => same(s, m))).length;

/* ------------------------------------------------------------ the steps */

type Step = {
  key: string;
  short: string;
  run: string;
  title: string;
  tried: ReactNode;
  found: ReactNode;
  why: ReactNode;
  stat: string;
  statSub: string;
  /** The stat is a result that holds, so it takes the accent. */
  holds?: boolean;
  visual: ReactNode;
  href: string;
  open: string;
};

const PHASES = [
  { name: "Replay", from: 0 },
  { name: "Scale", from: 2 },
  { name: "New memories", from: 4 },
  { name: "Confidence", from: 6 },
];

export const STEPS: Step[] = [
  {
    key: "raw",
    short: "Raw replay",
    run: "Run A",
    title: "The paper's replay, on real photos.",
    tried: <>The paper's equations, unchanged, on four photos encoded by I-JEPA: 1,280 numbers per photo, β = 3, γ = 0.2, δ = 0.1.</>,
    found: (
      <>
        One memory takes {pct1(winnerShare)} of the replay and is the only one left, 1 of 4. With ten photos,{" "}
        {ten.survivors_above_0_001.unnormalised} of {ten.memories.length}.
      </>
    ),
    why: (
      <>
        Replay scores are raw dot products. In the replay-tracking run they started near 640 for the top memory and 625 for the next.
        Times β = 3 that gap is 45, and the softmax puts all of its weight on the top score.
      </>
    ),
    stat: "1 of 4",
    statSub: `${pct1(winnerShare)} of replay to one memory`,
    visual: <MiniBars finals={unnormalisedFinals} scale={0.15} />,
    href: "#/unnormalised",
    open: "Open run A",
  },
  {
    key: "unit",
    short: "Unit-length replay",
    run: "Run B",
    title: "One change: unit-length embeddings.",
    tried: <>Scale each embedding to length 1 inside the replay operator only, so each score is a sum of cosine similarities.</>,
    found: (
      <>
        Replay splits between {pct1(lateShareRange[0])} and {pct1(lateShareRange[1])} each, and all four memories stay, in order of
        decay rate. With ten photos, {ten.survivors_above_0_001.normalised} of {ten.memories.length}.
      </>
    ),
    why: (
      <>
        A score can now be no larger than the total activation, so β = 3 no longer turns small differences into an all-or-nothing
        choice. The angles between memories stay; only the lengths go.
      </>
    ),
    stat: `${normalised.survivors_above_0_001.normalised} of 4`,
    statSub: `${ten.survivors_above_0_001.normalised} of ${ten.memories.length} with ten photos`,
    holds: true,
    visual: <MiniBars finals={normalisedFinals} scale={0.15} />,
    href: "#/normalised",
    open: "Open run B",
  },
  {
    key: "scale",
    short: "Paper scale",
    run: "Run C · test 1",
    title: `The paper's scale: ${paperScale.survivors} of ${ps.n}.`,
    tried: (
      <>
        The paper's N = {ps.n} and {ps.steps.toLocaleString("en")} steps, on {ps.n} CIFAR-100 images through I-JEPA, with run B's
        replay and the paper's γ and δ.
      </>
    ),
    found: (
      <>
        None of the {ps.n} stays above {ps.alive_threshold}. The order still follows decay rate, ρ ={" "}
        {paperScale.spearman_rho_decay_vs_activation.toFixed(3)}.
      </>
    ),
    why: (
      <>
        Inhibition takes δ times the total off every memory, so the total loses δN = {Math.round(deltaN)} times itself per unit of
        time, against decay of 0.5 to 1.5. Replay can hold it near ηp / (1 + δN) ≈ {holdTotal.toFixed(3)}, about{" "}
        {(holdTotal / ps.n).toFixed(5)} per memory. Similarity doesn't enter that sum.
      </>
    ),
    stat: `${paperScale.survivors} of ${ps.n}`,
    statSub: `above ${ps.alive_threshold} at step ${ps.steps.toLocaleString("en")}`,
    visual: <ThresholdMeter />,
    href: "#/paper",
    open: "Open run C",
  },
  {
    key: "sweep",
    short: "γ and δ sweep",
    run: "Run C · test 2",
    title: "Inhibition is the lever, not diffusion.",
    tried: <>{paperTuned.grid.length} runs over five values of γ and five of δ, everything else as in test 1.</>,
    found: (
      <>
        Only δ = {lowDelta} keeps any memories, and lowering γ alone changes nothing. γ = {best.gamma}, δ = {best.delta} keeps{" "}
        {best.alive} of {ps.n}, ρ = {best.rho.toFixed(3)}.
      </>
    ),
    why: (
      <>
        δ sets the level the total can hold. At δ = {lowDelta} the average memory sits right on the {ps.alive_threshold} line, so{" "}
        {best.alive} of {ps.n} is the count at step {ps.steps.toLocaleString("en")}, not a margin.
      </>
    ),
    stat: `${best.alive} of ${ps.n}`,
    statSub: `δ = ${ps.delta} → ${best.delta}`,
    holds: true,
    visual: <SurvivorGrid />,
    href: "#/paper/tuned",
    open: "Open the sweep",
  },
  {
    key: "midrun",
    short: "Mid-run memories",
    run: "Beyond the paper · test 1",
    title: "New memories, halfway through.",
    tried: (
      <>
        The paper stores every memory at step 0. Here five are stored at step 0 and five more added at step {fg.inject_at}, at
        activation {fg.new_memory_start_activation}, with run B's replay.
      </>
    ),
    found: (
      <>
        All four old photos stay; the one old memory lost is a random vector that stood in for a dead photo link. The old photos fall{" "}
        {Math.round(Math.min(...drops))}% to {Math.round(Math.max(...drops))}% after step {fg.inject_at}.
      </>
    ),
    why: (
      <>
        Five new memories at {fg.new_memory_start_activation} multiply the summed activation about {Math.round(sumAfter / sumBefore)}{" "}
        times, and inhibition presses with that sum. There is no copy of the run without them, so how much of the fall they cause isn't
        measured.
      </>
    ),
    stat: `${forgetting.survivors.old} of ${forgetting.survivors.old_total}`,
    statSub: "old memories left at step 500",
    visual: <Timeline />,
    href: "#/beyond",
    open: "Open test 1",
  },
  {
    key: "gentle",
    short: "Adding them gently",
    run: "Beyond the paper · test 2",
    title: "Let new memories in at the same volume.",
    tried: (
      <>
        The same test with a photo in every slot, run three ways: new memories start at 1, at the mean activation, or warm up over{" "}
        {fixes.setup.warmup_steps} steps.
      </>
    ),
    found: (
      <>
        Starting at the mean keeps {ret.mean}% of the old activation, against {ret.raw}% starting at 1; warming up keeps {ret.warmup}%.
        No memory drops below {fixes.setup.alive_threshold} in any of them.
      </>
    ),
    why: (
      <>
        Starting at the mean adds less to the sum inhibition presses with. Decay alone would keep {pct1(decayRange[0])} to{" "}
        {pct1(decayRange[1])} over the same steps, so the loss that's left is within ordinary decay. Scaling δ with the number of
        memories is the next fix to run.
      </>
    ),
    stat: `${ret.mean}%`,
    statSub: `of old activation kept, from ${ret.raw}%`,
    holds: true,
    visual: <RetentionBars />,
    href: "#/beyond/fix",
    open: "Open test 2",
  },
  {
    key: "bayes",
    short: "Bayes' rule",
    run: "Bayesian confidence · tests 1 and 2",
    title: "Bayes' rule, and every confidence falls.",
    tried: (
      <>
        Give each memory a confidence, starting at 0.5, updated at every replay by Bayes' rule with its replay weight r as the
        likelihood, and fed back into replay.
      </>
    ),
    found: (
      <>
        Every confidence ends at the {bayesFour.bayes.floor} floor: all {bayesFour.setup.n} by about step 370, all {bayes200.setup.n} by
        about step 70.
      </>
    ),
    why: (
      <>
        Bayes' rule multiplies a memory's odds by r/(1 − r), below 1 unless it gets more than half of a replay. With N memories r
        averages 1/N, so every replay is evidence against every memory.
      </>
    ),
    stat: bayesFour.bayes.floor.toFixed(2),
    statSub: `every confidence, N = ${bayesFour.setup.n} and N = ${bayes200.setup.n}`,
    visual: <ConfidenceRanges rows={CONFIDENCE_RANGES.slice(0, 2)} />,
    href: "#/bayes",
    open: "Open tests 1 and 2",
  },
  {
    key: "relative",
    short: "Relative evidence",
    run: "Bayesian confidence · test 3",
    title: "Count evidence against the average.",
    tried: (
      <>
        Count evidence as how many standard deviations a memory's replay weight sits above or below the average. Confidence is the
        sigmoid of the running sum.
      </>
    ),
    found: (
      <>
        Confidence spreads from 0 to 1 and has nothing to do with decay rate, ρ = {neg(bayesRelative.confidence_decay_rho.toFixed(3))}.
        The {word(topMatch)} most confident are the {word(topMatch)} that test 2's softer replay held back most.
      </>
    ),
    why: (
      <>
        Both measure whether a memory gets more replay than average. By the replay equation that depends mostly on how similar a memory
        is to all the others. The notebook doesn't print that, so it's a reading, not yet a measurement.
      </>
    ),
    stat: "0 → 1",
    statSub: `confidence, mean ${bayesRelative.confidence.mean.toFixed(3)}`,
    holds: true,
    visual: <ConfidenceRanges />,
    href: "#/bayes/relative",
    open: "Open test 3",
  },
];

/* ------------------------------------------------------------ pieces */

function StepText({ s, i }: { s: Step; i: number }) {
  return (
    <>
      <div className="label">
        {String(i + 1).padStart(2, "0")} · {s.run}
      </div>
      <h4 className="journey-title">{s.title}</h4>
      <dl className="journey-tfw">
        <div>
          <dt>Tried</dt>
          <dd>{s.tried}</dd>
        </div>
        <div>
          <dt>Found</dt>
          <dd>{s.found}</dd>
        </div>
        <div>
          <dt>Why</dt>
          <dd>{s.why}</dd>
        </div>
      </dl>
      <a className="journey-open" href={s.href}>
        {s.open} <span aria-hidden="true">→</span>
      </a>
    </>
  );
}

function StepVisual({ s }: { s: Step }) {
  return (
    <div className="journey-card">
      <div className={`stat-value${s.holds ? " ember" : ""}`}>{s.stat}</div>
      <div className="stat-sub">{s.statSub}</div>
      <div className="journey-chart">{s.visual}</div>
    </div>
  );
}

/** Pinned: the steps as a rail on the left, the current step's text in the middle, its run's chart on the right. */
function JourneyScrolly() {
  const [ref, p] = usePinnedProgress<HTMLDivElement>();
  const n = STEPS.length;
  const active = Math.min(n - 1, Math.floor(p * n * 1.0001));
  const go = (i: number) => {
    const el = ref.current;
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY;
    const travel = el.offsetHeight - window.innerHeight;
    window.scrollTo({ top: top + (travel * (i + 0.5)) / n });
  };
  return (
    <div className="journey-scrolly" ref={ref} style={{ "--n": n } as CSSProperties}>
      <div className="scrolly-sticky">
        <div className="wrap grid journey-grid">
          <nav className="journey-rail" aria-label="Steps of the research journey">
            <span className="jr-track" aria-hidden="true">
              <span className="jr-fill" style={{ height: `${Math.min(1, p) * 100}%` }} />
            </span>
            {PHASES.map((ph, k) => (
              <div key={ph.name} className="jr-phase">
                <div className="label">{ph.name}</div>
                <ol start={ph.from + 1}>
                  {STEPS.slice(ph.from, PHASES[k + 1]?.from ?? n).map((s, j) => {
                    const i = ph.from + j;
                    return (
                      <li key={s.key}>
                        <button
                          type="button"
                          className={`${i === active ? "on" : ""}${i < active ? " done" : ""}`}
                          aria-current={i === active ? "step" : undefined}
                          onClick={() => go(i)}
                        >
                          <span className="jr-num">{String(i + 1).padStart(2, "0")}</span>
                          <span className="jr-name">{s.short}</span>
                        </button>
                      </li>
                    );
                  })}
                </ol>
              </div>
            ))}
          </nav>
          <div className="journey-text">
            {STEPS.map((s, i) => (
              <div key={s.key} className={`journey-step${i === active ? " on" : ""}`} aria-hidden={i !== active}>
                <StepText s={s} i={i} />
              </div>
            ))}
          </div>
          <div className="journey-art">
            {STEPS.map((s, i) => (
              <div key={s.key} className={`journey-step${i === active ? " on" : ""}`} aria-hidden={i !== active}>
                <StepVisual s={s} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Narrow screens: the same steps as a list, each with its chart. */
function JourneyList() {
  return (
    <div className="wrap">
      <ol className="journey-list">
        {STEPS.map((s, i) => (
          <li key={s.key} data-reveal>
            <StepText s={s} i={i} />
            <StepVisual s={s} />
          </li>
        ))}
      </ol>
    </div>
  );
}

const SUMMARY: { step: string; href: string; did: ReactNode; result: ReactNode; kind: string }[] = [
  {
    step: "Raw replay on real embeddings",
    href: "#/unnormalised",
    did: "The paper's equations on I-JEPA, 4 and 10 photos",
    result: `One memory takes all replay: 1 of 4, ${ten.survivors_above_0_001.unnormalised} of ${ten.memories.length}`,
    kind: "Gap in the paper's setup",
  },
  {
    step: "Unit-length replay",
    href: "#/normalised",
    did: "Embeddings scaled to length 1 inside F(a)",
    result: `${normalised.survivors_above_0_001.normalised} of 4 and ${ten.survivors_above_0_001.normalised} of ${ten.memories.length}, in decay order`,
    kind: "Fix",
  },
  {
    step: "The paper's scale",
    href: "#/paper",
    did: `N = ${ps.n}, T = ${ps.steps.toLocaleString("en")}, the paper's γ and δ`,
    result: `${paperScale.survivors} of ${ps.n} above ${ps.alive_threshold}`,
    kind: "Gap in the paper's setup",
  },
  {
    step: "γ and δ sweep",
    href: "#/paper/tuned",
    did: `${paperTuned.grid.length} runs, five values of each`,
    result: `Only δ = ${lowDelta} keeps any; ${best.alive} of ${ps.n} at γ = ${best.gamma}`,
    kind: "Fix",
  },
  {
    step: "New memories mid-run",
    href: "#/beyond",
    did: `Five more at step ${fg.inject_at}, at activation ${fg.new_memory_start_activation}`,
    result: "All old photos stay; their activation falls",
    kind: "Beyond the paper",
  },
  {
    step: "Adding them gently",
    href: "#/beyond/fix",
    did: "New memories start at the mean, or warm up",
    result: `${ret.mean}% of old activation kept, against ${ret.raw}%`,
    kind: "Beyond the paper",
  },
  {
    step: "Bayes' rule confidence",
    href: "#/bayes",
    did: `Replay weight as the likelihood, N = ${bayesFour.setup.n} and ${bayes200.setup.n}`,
    result: `Every confidence at the ${bayesFour.bayes.floor} floor`,
    kind: "Beyond the paper",
  },
  {
    step: "Relative evidence",
    href: "#/bayes/relative",
    did: "Standardised evidence, then a sigmoid",
    result: `Confidence 0 to 1, ρ = ${neg(bayesRelative.confidence_decay_rho.toFixed(3))} against decay`,
    kind: "Beyond the paper",
  },
  {
    step: "What sets confidence",
    href: "#/bayes/relative",
    did: "Compared with the shifts in test 2",
    result: "The same memories: each one's share of replay",
    kind: "Open: similarity not yet measured",
  },
];

/* ------------------------------------------------------------ the chapter */

/** Home: the runs told in order, as a pinned scroll on wide screens and a list on narrow ones, then a summary table. */
export function JourneyChapter({ num }: { num: number }) {
  const narrow = useIsNarrow();
  return (
    <>
      <Chapter
        id="journey"
        sub={
          <div className="grid term-row">
            {PHASES.map((ph) => (
              <span key={ph.name} className="label ink">
                {ph.name}
              </span>
            ))}
          </div>
        }
      >
        Journey
      </Chapter>
      <Section
        id="research-journey"
        label="Research journey"
        num={num}
        headline="What I tried, what broke, and what fixed it."
        lead="Eight steps, one run each. Each says what was tried, what happened and why, with that run's recorded numbers. The model above is where it starts."
      />
      {narrow ? <JourneyList /> : <JourneyScrolly />}
      <Section
        id="journey-summary"
        label="In one table"
        num={num + 1}
        headline="Two gaps, two fixes, two new questions."
        lead="The paper's equations hold on real I-JEPA embeddings once replay uses unit-length embeddings and inhibition is turned down for 200 memories. Beyond the paper, new memories can join mid-run without any old photo dropping below 0.001, and a confidence counted against the average replay separates memories by their share of replay, not by how fast they decay."
      >
        <div className="grid">
          <div className="indent stack">
            <Card label="Every step" meta="Each links to its run">
              <div className="scroll">
                <table className="journey-table">
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>What was done</th>
                      <th>Result</th>
                      <th>Kind</th>
                    </tr>
                  </thead>
                  <tbody>
                    {SUMMARY.map((r) => (
                      <tr key={r.step}>
                        <td>
                          <a href={r.href}>{r.step}</a>
                        </td>
                        <td>{r.did}</td>
                        <td>{r.result}</td>
                        <td className="muted">{r.kind}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </div>
      </Section>
    </>
  );
}
