import { Fragment, type ReactNode } from "react";
import { bayes200, bayesFour, bayesRelative, figures, NOTEBOOKS, paperTuned, PHOTOS } from "../data/results";
import { file } from "../data/derived";
import { Card, Section } from "../components/Layout";
import { RunIntro } from "../components/RunIntro";
import { Tex } from "../components/Tex";
import { N } from "../components/ui";

type Test = 1 | 2 | 3;

/** A CIFAR-100 class name as text: "maple_tree" reads "maple tree". */
const cls = (label: string) => label.replace(/_/g, " ");
const sig = (x: number) => 1 / (1 + Math.exp(-x));

/* ------------------------------------------------------------ worked out from the notebooks' settings */

const s4 = bayesFour.setup;
const k4 = bayesFour.bayes;
const s2 = bayes200.setup;
const k2 = bayes200.bayes;
const s3 = bayesRelative.setup;
const k3 = bayesRelative.bayes;

/** Share of confidence that the decay term alone keeps over a run: (1 − λh)^T. */
const decayKeep = (lambda: number, h: number, steps: number) => (1 - lambda * h) ** steps;
/** The weight a confidence c puts on its memory's activation inside the replay scores. */
const cWeight = (w: number, c: number) => 1 - w + w * c;
/**
 * Replay events for the Bayes'-rule update to take a confidence from the prior to the floor,
 * if the memory's replay weight were r at every event.
 */
function eventsToFloor(r: number, kappa: number, prior: number, floor: number) {
  let c = prior;
  let n = 0;
  while (c > floor && n < 100000) {
    const post = (r * c) / (r * c + (1 - r) * (1 - c));
    c += kappa * (post - c);
    n += 1;
  }
  return n;
}

const keep4 = decayKeep(k4.confidence_decay_rate, s4.h, s4.steps);
const kStart4 = cWeight(k4.confidence_weight, k4.prior_confidence);
const kFloor4 = cWeight(k4.confidence_weight, k4.floor);
const events4 = eventsToFloor(1 / s4.n, k4.confidence_boost, k4.prior_confidence, k4.floor);

const r200 = 1 / s2.n;
const keep200 = decayKeep(k2.confidence_decay_rate, s2.h, s2.steps);
const events200 = eventsToFloor(r200, k2.confidence_boost, k2.prior_confidence, k2.floor);
const post200 = (r200 * k2.prior_confidence) / (r200 * k2.prior_confidence + (1 - r200) * (1 - k2.prior_confidence));
const kFloor200 = cWeight(k2.confidence_weight, k2.floor);
const maxShift200 = Math.max(...[...bayes200.boosted_top5, ...bayes200.suppressed_top5].map((m) => Math.abs(m.diff)));
const maxShiftRow = [...bayes200.boosted_top5, ...bayes200.suppressed_top5].find((m) => Math.abs(m.diff) === maxShift200)!;
const runCCell = paperTuned.grid.find((g) => g.gamma === s2.gamma && g.delta === s2.delta);

const events3 = Math.round(s3.steps * s3.replay_prob);
const evidence9 = k3.conf_lr * events3;
const keepEvidence = decayKeep(k3.conf_decay, s3.h, s3.steps);
const total3 = s3.n * bayesRelative.standard.mean;
const shareRatioBound = Math.exp(s3.beta * total3);
const same = (a: { label: string; decay: number }, b: { label: string; decay: number }) => a.label === b.label && a.decay === b.decay;
const topInTest2 = bayesRelative.most_confident_top5.filter((m) => bayes200.suppressed_top5.some((s) => same(s, m))).length;
const bottomInTest2 = bayesRelative.least_confident_top5.filter((m) => bayes200.boosted_top5.some((s) => same(s, m))).length;
const lion = bayesRelative.least_confident_top5.reduce((a, b) => (b.decay < a.decay ? b : a));
const trusted = bayesRelative.most_confident_top5.reduce((a, b) => (b.decay < a.decay ? b : a));

const f4 = (v: number) => v.toFixed(4);
const f6 = (v: number) => v.toFixed(6);
const pct = (v: number) => `${Math.round(v * 100)}%`;
/** A printed number with a typographic minus. */
export const neg = (v: number | string) => String(v).replace("-", "−");
const confRho = neg(bayesRelative.confidence_decay_rho);
const word = (n: number) => ["no", "one", "two", "three", "four", "five"][n] ?? String(n);

/* ------------------------------------------------------------ page */

/** Beyond the paper: a confidence for every memory, updated from replay, in three tests. */
export function RunBayes({ test }: { test: Test }) {
  return (
    <>
      <RunIntro
        run="Beyond the paper"
        name="Bayesian confidence"
        headline="A second number for every memory: how much to trust it."
        explanation={
          <>
            <p>
              The paper gives each memory one number, its activation. These tests add a second, a confidence c between 0 and 1. It
              starts at 0.5, is updated at every replay event from the share of that replay the memory got, and feeds back into replay:
              a memory's activation counts for less in the replay scores when its confidence is low.
            </p>
            <p>
              Tests 1 and 2 update confidence with Bayes' rule, taking the replay weight as the likelihood, on run B's four photos and
              then on run C's 200 images. In both, every confidence ends at the floor. Test 3 counts evidence relative to the average
              instead, and confidence spreads from 0 to 1. Pick one.
            </p>
          </>
        }
        formula={String.raw`F(a, c) = \mathrm{softmax}\big(\beta\, \hat X^\top \hat X\, (a \odot (1 - w + w\,c))\big)`}
        formulaNote={
          <>
            Run B's normalised replay with each activation weighted by its confidence. <N>w</N> = {k4.confidence_weight} in tests 1
            and 2, {k3.conf_weight} in test 3. With c = 1 for every memory this is run B's replay.
          </>
        }
      />
      <BayesTestCards test={test} />
      {test === 1 ? <Test1View /> : test === 2 ? <Test2View /> : <Test3View />}
    </>
  );
}

type Range = { name: string; min: number; max: number; mean: number; fix?: boolean };

export const CONFIDENCE_RANGES: Range[] = [
  { name: "Test 1 · 4 photos", min: bayesFour.memories[0].confidence, max: bayesFour.memories[0].confidence, mean: bayesFour.memories[0].confidence },
  {
    name: "Test 2 · 200 images",
    min: bayes200.bayesian.confidence.min,
    max: bayes200.bayesian.confidence.max,
    mean: bayes200.bayesian.confidence.mean,
  },
  {
    name: "Test 3 · relative",
    min: bayesRelative.confidence.min,
    max: bayesRelative.confidence.max,
    mean: bayesRelative.confidence.mean,
    fix: true,
  },
];

const rangeText = (r: Range) => (r.max - r.min < 1e-9 ? r.min.toFixed(2) : `${r.min.toFixed(0)} to ${r.max.toFixed(0)}`);

/** Final confidences in each test on a 0 to 1 scale: the span from lowest to highest, the mean, and the starting 0.5. */
export function ConfidenceRanges({ rows = CONFIDENCE_RANGES, single = false }: { rows?: Range[]; single?: boolean }) {
  return (
    <ul
      className={`conf-ranges${single ? " single" : ""}`}
      aria-label={rows.map((r) => `${r.name}: final confidence ${rangeText(r)}, mean ${r.mean.toFixed(2)}`).join("; ")}
    >
      {rows.map((r) => (
        <li key={r.name} className={r.fix ? "fix" : undefined}>
          {!single && <span className="cr-name">{r.name}</span>}
          <span className="cr-track">
            <span className="cr-prior" style={{ left: "50%" }} />
            <span className="cr-fill" style={{ left: `${r.min * 100}%`, width: `${(r.max - r.min) * 100}%` }} />
            {r.max - r.min > 0.05 && <span className="cr-mean" style={{ left: `${r.mean * 100}%` }} />}
          </span>
          <span className="cr-value">{rangeText(r)}</span>
        </li>
      ))}
      <li className="cr-axis" aria-hidden="true">
        {!single && <span />}
        <span className="cr-ticks">
          <span>0</span>
          <span>0.5 start</span>
          <span>1</span>
        </span>
        <span />
      </li>
    </ul>
  );
}

function BayesTestCards({ test }: { test: Test }) {
  const card = (n: Test, href: string, title: string, desc: string, stats: [string, ReactNode][]) => (
    <a className={`run-card experiment-card${test === n ? " on" : ""}`} href={href} aria-current={test === n ? "page" : undefined}>
      <div className="run-card-head">
        <span className="label">Test {n}</span>
        {test === n && <span className="tag new">Showing</span>}
      </div>
      <h2 className="run-title">{title}</h2>
      <p className="run-desc">{desc}</p>
      <ConfidenceRanges rows={[CONFIDENCE_RANGES[n - 1]]} single />
      <dl className="run-stats">
        {stats.map(([dt, dd]) => (
          <div key={dt}>
            <dt>{dt}</dt>
            <dd>{dd}</dd>
          </div>
        ))}
      </dl>
    </a>
  );
  return (
    <section className="section" id="bayes-tests">
      <div className="wrap">
        <div className="run-cards bayes-cards">
          {card(
            1,
            "#/bayes",
            "Bayes' rule, four photos",
            `Run B's four photos for ${s4.steps} steps, with the replay weight as the likelihood. Every confidence falls from 0.5 to the ${k4.floor} floor.`,
            [
              ["Final confidence", <>{f4(bayesFour.memories[0].confidence)} <span className="dd-sub">all four</span></>],
              ["Memories left", <>{bayesFour.survivors.bayesian} of {s4.n} <span className="dd-sub">in both runs</span></>],
            ],
          )}
          {card(
            2,
            "#/bayes/200",
            "Bayes' rule, 200 images",
            `Run C's ${s2.n} images for ${s2.steps.toLocaleString("en")} steps, with a bigger update step and slower decay. Confidence reaches the floor faster.`,
            [
              ["Final confidence", <>{f4(bayes200.bayesian.confidence.mean)} <span className="dd-sub">all {s2.n}</span></>],
              ["Order by decay", <>ρ {bayes200.standard.rho.toFixed(3)} → {bayes200.bayesian.rho.toFixed(3)}</>],
            ],
          )}
          {card(
            3,
            "#/bayes/relative",
            "Relative evidence",
            `The same ${s3.n} images, with evidence counted against the average replay weight. Confidence spreads from 0 to 1.`,
            [
              ["Mean confidence", <>{bayesRelative.confidence.mean.toFixed(3)} <span className="dd-sub">from 0.0000 to 1.0000</span></>],
              ["Confidence vs decay", <>ρ = {neg(bayesRelative.confidence_decay_rho.toFixed(3))}</>],
            ],
          )}
        </div>
      </div>
    </section>
  );
}

/** Thumbnail of one of run B's photos. */
function Thumb({ label }: { label: string }) {
  return <img className="thumb" src={PHOTOS[label].thumb} alt="" width={48} height={36} />;
}

/** Shows / Doesn't show, in two columns. */
function Findings({ num, shows, not }: { num: number; shows: ReactNode[]; not: ReactNode[] }) {
  return (
    <Section id="findings" label="Findings" num={num} headline="What it shows, and what it doesn't.">
      <div className="grid">
        <div className="indent two-col">
          <div data-reveal>
            <div className="label">Shows</div>
            <ul className="points">
              {shows.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
          <div data-reveal>
            <div className="label">Doesn't show</div>
            <ul className="points">
              {not.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
}

function CodeSection({
  num,
  notebook,
  path,
  output,
  json,
  links,
}: {
  num: number;
  notebook: string;
  path: string;
  output: string;
  json: string;
  links: ReactNode;
}) {
  return (
    <Section id="run-code" label="The code" num={num} headline="Where these numbers come from.">
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>The executed notebook.</b> <a href={notebook}>{path}</a>, with its code, printed output and figure as they ran in Colab.
            </li>
            <li>
              <b>Recorded copies.</b> Printed output in <a href={file(output)}>{output}</a>, numbers in <a href={file(json)}>{json}</a>.
            </li>
          </ul>
          <div className="btn-row">
            <a className="btn btn-primary" href={notebook}>
              Open the Colab notebook on GitHub
            </a>
            {links}
          </div>
        </div>
      </div>
    </Section>
  );
}

function FigureSection({ num, headline, label, src, alt, note }: { num: number; headline: string; label: string; src: string; alt: string; note: ReactNode }) {
  return (
    <Section id="figure" label="Figure" num={num} headline={headline}>
      <div className="grid">
        <div className="indent stack">
          <Card label={label} meta="Colab output" note={note}>
            <div className="plate">
              <img src={src} alt={alt} />
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ test 1 */

function Test1View() {
  return (
    <>
      <Setup1 num={1} />
      <Update1 num={2} />
      <Result1 num={3} />
      <Why1 num={4} />
      <Softer1 num={5} />
      <FigureSection
        num={6}
        headline="Both runs, step by step."
        label="Figure · Standard against Bayesian, 4 memories"
        src={figures.bayesFour}
        alt="Six panels. Top: activations of the four memories in the standard and Bayesian runs, nearly identical, falling from 1 and levelling off between 0.03 and 0.1. Middle left: a text box, since the standard model has no confidence. Middle right: all four confidences fall together in small steps from 0.5 to 0.01 by about step 370. Bottom left: final activations side by side. Bottom right: the Bayesian run's activations, its confidences of 0.01, and activation times confidence near zero."
        note={
          <>
            The notebook's own figure. Top: activations, standard on the left and Bayesian on the right. Middle right: confidence over
            time, with the starting 0.5 dashed. Bottom left: final activations side by side. Bottom right: the Bayesian run's final
            activation, confidence and activation × confidence.
          </>
        }
      />
      <Findings
        num={7}
        shows={[
          <>With the replay weight as the likelihood, Bayes' rule lowers every memory's confidence at every replay, and all four end at the {k4.floor} floor.</>,
          <>Confidence that is the same for every memory only softens replay: the most-replayed memory loses a little and the others gain a little.</>,
          <>All four memories stay above {s4.alive_threshold}, in the same order, in both runs.</>,
        ]}
        not={[
          <>Anything about how reliable a memory is. Confidence here depends only on replay weights.</>,
          <>That a standard run with <N>β</N> ≈ {(s4.beta * kFloor4).toFixed(1)} gives the Bayesian run's numbers: that run wasn't done.</>,
          <>Anything statistical: one seed.</>,
        ]}
      />
      <CodeSection
        num={8}
        notebook={NOTEBOOKS.bayesFour}
        path="notebooks/colab/baysian approach/baysian_4_mem.ipynb"
        output="results/bayesian_4_memories_output.txt"
        json="results/bayesian_4_memories.json"
        links={
          <a className="btn btn-text" href="#/bayes/200">
            Next: test 2, the same rule on 200 images <span aria-hidden="true">→</span>
          </a>
        }
      />
    </>
  );
}

function Setup1({ num }: { num: number }) {
  return (
    <Section
      id="test"
      label="The test"
      num={num}
      headline="Run B's four photos, with a confidence for each."
      lead={`Two runs from the same start: the standard model with normalised replay, and the same model with a confidence per memory. Both use seed ${s4.seed} and draw one random number a step, so replay fires on the same steps in both.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Setup"
            meta="baysian_4_mem.ipynb"
            note={
              <>
                Decay rates here are evenly spaced, <code>torch.linspace(0.5, 1.5, 4)</code>. Run B drew its decay rates at random, so the
                standard run's numbers on this page are not run B's.
              </>
            }
          >
            <div className="scroll">
              <table className="thumb-table">
                <thead>
                  <tr>
                    <th>Memory</th>
                    <th className="num">
                      <N>α</N>
                    </th>
                    <th className="num">Embedding length</th>
                  </tr>
                </thead>
                <tbody>
                  {bayesFour.memories.map((m) => (
                    <tr key={m.label}>
                      <td>
                        <span className="thumb-name">
                          <Thumb label={m.label} />
                          <span>{PHOTOS[m.label].name}</span>
                        </span>
                      </td>
                      <td className="num">{m.decay_rate.toFixed(2)}</td>
                      <td className="num">{bayesFour.embedding_norms_printed[m.label]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="points compact" style={{ marginTop: 18 }}>
              <li>
                <b>Both runs:</b> <N>γ</N> = {s4.gamma}, <N>δ</N> = {s4.delta}, <N>β</N> = {s4.beta}, <N>η</N> = {s4.eta}, p ={" "}
                {s4.replay_prob}, h = {s4.h}, {s4.steps} steps, seed {s4.seed}.
              </li>
              <li>
                <b>Confidence:</b> starts at {k4.prior_confidence}; update step <N>κ</N> = {k4.confidence_boost}; decay <N>λ</N> ={" "}
                {k4.confidence_decay_rate}; weight in replay <N>w</N> = {k4.confidence_weight}; kept between {k4.floor} and {k4.ceiling}.
              </li>
            </ul>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function Update1({ num }: { num: number }) {
  return (
    <Section
      id="update"
      label="The update"
      num={num}
      headline="Bayes' rule, with the replay weight as the likelihood."
      lead="At every replay event, each memory's confidence moves part of the way toward a posterior worked out from r, the share of that replay it got. Every step it also decays a little. These are the notebook's lines, written out."
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="Confidence update" meta="run_bayesian">
            <Tex block>{String.raw`\tilde c_i = \frac{r_i\, c_i}{r_i\, c_i + (1 - r_i)(1 - c_i)}, \qquad c_i \leftarrow c_i + \kappa\,(\tilde c_i - c_i)`}</Tex>
            <Tex block>{String.raw`c_i \leftarrow \mathrm{clip}\big(c_i - \lambda h\, c_i,\; ${k4.floor},\; ${k4.ceiling}\big) \quad \text{every step}`}</Tex>
            <p className="card-note" style={{ marginTop: 4 }}>
              <N>κ</N> = {k4.confidence_boost} takes {pct(k4.confidence_boost)} of the step to the posterior. The replay weights r come
              from the confidence-weighted replay above, so confidence feeds back on itself. The notebook's comments write the replay as{" "}
              <Tex>{String.raw`a \odot c`}</Tex>; the code blends, so a confidence of 0 still leaves 1 − <N>w</N> ={" "}
              {1 - k4.confidence_weight} of the activation.
            </p>
          </Card>
          <Card label="The same rule, in odds" meta="Worked out from the update">
            <Tex block>{String.raw`\frac{\tilde c_i}{1 - \tilde c_i} \;=\; \frac{r_i}{1 - r_i} \cdot \frac{c_i}{1 - c_i}`}</Tex>
            <p className="card-note" style={{ marginTop: 4 }}>
              Bayes' rule multiplies a memory's odds by r/(1 − r). That is above 1 only when the memory gets more than half of a replay.
              The weights add up to 1, so at most one memory can. With four memories sharing replay, each gets a quarter on average, and
              r/(1 − r) = 1/3: every replay is evidence against every memory.
            </p>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function Result1({ num }: { num: number }) {
  const beagle = bayesFour.memories[0];
  return (
    <Section
      id="result"
      label="The result"
      num={num}
      headline={`Every confidence ends at the ${k4.floor} floor.`}
      lead={`All four memories stay above ${s4.alive_threshold} in both runs, in the same order. Confidence prints ${f4(beagle.confidence)} for every memory, the lowest value the code allows.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Final confidence" meta="Every memory">
              <div className="stat-value">{f4(beagle.confidence)}</div>
              <div className="stat-sub">The floor. It started at {k4.prior_confidence}.</div>
            </Card>
            <Card label="Memories left" meta={`Above ${s4.alive_threshold}`} delay={1}>
              <div className="stat-value">
                {bayesFour.survivors.bayesian}
                <span className="unit">of {s4.n}</span>
              </div>
              <div className="stat-sub">In both runs, ordered by decay rate.</div>
            </Card>
            <Card label="Biggest change" meta={PHOTOS[beagle.label].name} delay={2}>
              <div className="stat-value ember">{neg(f4(beagle.difference_printed))}</div>
              <div className="stat-sub">
                Bayesian minus standard, at step {s4.steps}. The beagle decays slowest, <N>α</N> = {beagle.decay_rate.toFixed(2)}.
              </div>
            </Card>
          </div>
          <Card label={`Activation at step ${s4.steps}`} meta="As printed" note="Difference and activation × confidence are the notebook's own columns, from unrounded values.">
            <div className="scroll">
              <table className="thumb-table">
                <thead>
                  <tr>
                    <th>Memory</th>
                    <th className="num">
                      <N>α</N>
                    </th>
                    <th className="num">Standard</th>
                    <th className="num">Bayesian</th>
                    <th className="num">Difference</th>
                    <th className="num">Confidence</th>
                    <th className="num">a × c</th>
                  </tr>
                </thead>
                <tbody>
                  {bayesFour.memories.map((m) => (
                    <tr key={m.label}>
                      <td>
                        <span className="thumb-name">
                          <Thumb label={m.label} />
                          <span>{PHOTOS[m.label].name}</span>
                        </span>
                      </td>
                      <td className="num">{m.decay_rate.toFixed(2)}</td>
                      <td className="num">{f4(m.standard)}</td>
                      <td className="num big">{f4(m.bayesian)}</td>
                      <td className={`num${m.difference_printed < 0 ? " ember-text" : ""}`}>
                        {m.difference_printed > 0 ? "+" : ""}
                        {neg(f4(m.difference_printed))}
                      </td>
                      <td className="num">{f4(m.confidence)}</td>
                      <td className="num">{f4(m.effective_printed)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function Why1({ num }: { num: number }) {
  return (
    <Section
      id="why"
      label="Why"
      num={num}
      headline="The update empties confidence, not the decay."
      lead="Decay alone would barely have moved it. What brings it down is each replay event."
    >
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>Decay alone.</b> The decay step takes <N>λ</N>h = {k4.confidence_decay_rate} × {s4.h} ={" "}
              {(k4.confidence_decay_rate * s4.h).toFixed(4)} of the confidence a step. Over {s4.steps} steps that keeps (1 −{" "}
              {(k4.confidence_decay_rate * s4.h).toFixed(4)})<sup>{s4.steps}</sup> ≈ {pct(keep4)}, which would leave a confidence of 0.5
              near {(0.5 * keep4).toFixed(2)}.
            </li>
            <li>
              <b>Every replay is evidence against.</b> With r near a quarter, the posterior has a third of the prior's odds, and{" "}
              <N>κ</N> = {k4.confidence_boost} moves confidence {pct(k4.confidence_boost)} of the way there. If every weight were exactly
              1/4, getting from 0.5 to {k4.floor} would take {events4} replay events, about {Math.round(events4 / s4.replay_prob / 10) * 10}{" "}
              steps at p = {s4.replay_prob}. In the figure the four curves step down together and reach the floor at about step 370, read
              off the panel.
            </li>
            <li>
              <b>All four move together.</b> Their replay weights are close, so the update is close for each, and the four confidence
              curves in the figure lie almost on top of each other the whole way down.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function Softer1({ num }: { num: number }) {
  const beagle = bayesFour.memories[0];
  const gains = bayesFour.memories.slice(1).map((m) => m.difference_printed);
  return (
    <Section
      id="replay"
      label="Replay"
      num={num}
      headline="The same confidence for every memory just softens replay."
      lead={
        <>
          When every confidence is equal, the weight 1 − <N>w</N> + <N>w</N>c is one number k for every memory, and it comes out of the
          softmax as a smaller <N>β</N>.
        </>
      }
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="A common factor" meta="Worked out from the replay operator">
            <Tex block>{String.raw`\mathrm{softmax}\big(\beta\, \hat X^\top \hat X\, (k\,a)\big) \;=\; \mathrm{softmax}\big((k\beta)\, \hat X^\top \hat X\, a\big)`}</Tex>
          </Card>
          <ul className="points" data-reveal>
            <li>
              <b>
                From <N>β</N> = {s4.beta} to about {(s4.beta * kFloor4).toFixed(1)}.
              </b>{" "}
              k is {kStart4} while confidence is {k4.prior_confidence} and {kFloor4} at the floor. So the Bayesian run is run B's replay
              with <N>β</N> sliding from {(s4.beta * kStart4).toFixed(2)} to {(s4.beta * kFloor4).toFixed(3)}, where the standard run
              keeps {s4.beta}.
            </li>
            <li>
              <b>A smaller <N>β</N> spreads replay more evenly.</b> The beagle has the highest activation and so the highest replay score;
              it loses {Math.abs(beagle.difference_printed)}. The other three gain between {f4(Math.min(...gains))} and{" "}
              {f4(Math.max(...gains))}.
            </li>
            <li>
              <b>So confidence changes activations only through softer replay.</b> It never treats one memory differently from another,
              because it is the same for all four.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ test 2 */

function Test2View() {
  return (
    <>
      <Setup2 num={1} />
      <Result2 num={2} />
      <Why2 num={3} />
      <Moved2 num={4} />
      <FigureSection
        num={5}
        headline="All 200, step by step."
        label="Figure · Standard against Bayesian, 200 memories"
        src={figures.bayes200}
        alt="Six panels. Top: all 200 activations in each run fall from 1 to near 0 within about 500 steps. Middle left and bottom left: final activation against decay rate, every point above the 0.001 line, tighter in the Bayesian run. Middle right: every confidence falls from 0.5 to 0.01 within the first hundred steps and stays there. Bottom right: Bayesian minus standard activation for each memory, within about plus or minus 0.000015."
        note={
          <>
            The notebook's own figure. Top: activations, standard on the left and Bayesian on the right, the top ten highlighted. Middle
            left and bottom left: final activation against decay rate, standard then Bayesian, with the {s2.alive_threshold} line. Middle
            right: confidence over time, the five highest solid and the five lowest dashed, every one at {k2.floor}. Bottom right:
            Bayesian minus standard for each memory; it marks changes beyond ±0.00001.
          </>
        }
      />
      <Findings
        num={6}
        shows={[
          <>At N = {s2.n} the same rule empties confidence faster: about 70 steps here, about 370 in test 1.</>,
          <>A bigger update step and slower decay don't help, because the update itself lowers confidence whenever r is below 1/2.</>,
          <>
            Activations move by at most {maxShift200.toFixed(6)}, and the order by decay tightens, ρ {bayes200.standard.rho} →{" "}
            {bayes200.bayesian.rho}.
          </>,
        ]}
        not={[
          <>Any difference in confidence between memories: all {s2.n} end at {k2.floor}.</>,
          <>Which memories gain confidence when it can rise: it never does. That is test 3.</>,
          <>Anything statistical: one seed.</>,
        ]}
      />
      <CodeSection
        num={7}
        notebook={NOTEBOOKS.bayes200}
        path="notebooks/colab/baysian approach/baysian_200_mem.ipynb"
        output="results/bayesian_200_output.txt"
        json="results/bayesian_200.json"
        links={
          <>
            <a className="btn btn-text" href="#/bayes/relative">
              Next: test 3, relative evidence <span aria-hidden="true">→</span>
            </a>
            <a className="btn btn-text" href="#/bayes">
              <span aria-hidden="true">←</span> Test 1
            </a>
          </>
        }
      />
    </>
  );
}

function Setup2({ num }: { num: number }) {
  const rows: [ReactNode, ReactNode, ReactNode][] = [
    ["Memories", `${s4.n} photos`, `${s2.n} CIFAR-100 test images, mean cosine similarity ${s2.mean_cosine}`],
    ["Steps", s4.steps, s2.steps.toLocaleString("en")],
    [
      <>
        <N>γ</N>, <N>δ</N>
      </>,
      `${s4.gamma}, ${s4.delta}`,
      `${s2.gamma}, ${s2.delta}`,
    ],
    [
      <>
        Update step <N>κ</N>
      </>,
      k4.confidence_boost,
      k2.confidence_boost,
    ],
    [
      <>
        Confidence decay <N>λ</N>
      </>,
      k4.confidence_decay_rate,
      k2.confidence_decay_rate,
    ],
    [
      <>
        Weight in replay <N>w</N>
      </>,
      k4.confidence_weight,
      k2.confidence_weight,
    ],
  ];
  return (
    <Section
      id="test"
      label="The test"
      num={num}
      headline={`The same rule, on run C's ${s2.n} images.`}
      lead={`The ${s2.n} CIFAR-100 images from run C for ${s2.steps.toLocaleString("en")} steps, with run C test 2's γ = ${s2.gamma} and δ = ${s2.delta}, where all ${s2.n} stay. The notebook's notes read test 1's collapse as decay outrunning replay, so the confidence update gets a bigger step and a slower decay.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="What changed from test 1"
            meta="baysian_200_mem.ipynb"
            note={
              runCCell && (
                <>
                  The standard run is run C test 2's cell for <N>γ</N> = {s2.gamma}, <N>δ</N> = {s2.delta}, and prints the same numbers:{" "}
                  {bayes200.standard.survivors} of {s2.n} left, ρ = {bayes200.standard.rho}, highest {f6(bayes200.standard.max)} (run C:{" "}
                  {runCCell.alive}, {runCCell.rho}, {f6(runCCell.max)}).
                </>
              )
            }
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th />
                    <th>Test 1</th>
                    <th>Test 2</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([name, a, b], i) => (
                    <tr key={i}>
                      <td>{name}</td>
                      <td>{a}</td>
                      <td>{b}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="card-note" style={{ marginTop: 14 }}>
              The same in both: <N>β</N> = {s2.beta}, <N>η</N> = {s2.eta}, p = {s2.replay_prob}, h = {s2.h}, seed {s2.seed}, confidence
              starting at {k2.prior_confidence} and kept between {k2.floor} and {k2.ceiling}.
            </p>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function Result2({ num }: { num: number }) {
  const st = bayes200.standard;
  const by = bayes200.bayesian;
  const rows: [string, ReactNode, ReactNode][] = [
    [`Memories left (above ${s2.alive_threshold})`, st.survivors, by.survivors],
    ["Spearman ρ, decay against activation", st.rho.toFixed(4), by.rho.toFixed(4)],
    ["Highest activation", f6(st.max), f6(by.max)],
    ["Mean activation", f6(st.mean), f6(by.mean)],
    ["Mean confidence", "—", f4(by.confidence.mean)],
    ["Confidence spread (highest − lowest)", "—", f4(by.confidence.spread)],
  ];
  return (
    <Section
      id="result"
      label="The result"
      num={num}
      headline="Confidence hits the floor again, within about 70 steps."
      lead={`Every memory ends at ${f4(by.confidence.mean)}. The notebook's correlation between confidence and decay rate prints nan: with every confidence the same, there is nothing to rank.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Final confidence" meta={`All ${s2.n}`}>
              <div className="stat-value">{f4(by.confidence.mean)}</div>
              <div className="stat-sub">Spread {f4(by.confidence.spread)}. The floor is {k2.floor}.</div>
            </Card>
            <Card label="Memories left" meta={`Above ${s2.alive_threshold}`} delay={1}>
              <div className="stat-value">
                {by.survivors}
                <span className="unit">of {s2.n}</span>
              </div>
              <div className="stat-sub">In both runs.</div>
            </Card>
            <Card label="Order by decay" meta="Spearman ρ" delay={2}>
              <div className="stat-value ember">{by.rho}</div>
              <div className="stat-sub">Up from {st.rho} in the standard run.</div>
            </Card>
          </div>
          <Card label="The notebook's comparison" meta="As printed">
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th />
                    <th className="num">Standard</th>
                    <th className="num">Bayesian</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([name, a, b]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td className="num">{a}</td>
                      <td className="num">{b}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function Why2({ num }: { num: number }) {
  return (
    <Section
      id="why"
      label="Why"
      num={num}
      headline="A bigger step made the fall faster."
      lead="The update is what pushes confidence down, so strengthening it speeds the collapse. Slowing the decay changes nothing that matters."
    >
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>One full Bayes step goes almost to 0.</b> With {s2.n} memories each replay weight averages 1/{s2.n}, so r/(1 − r) is
              about 1/{s2.n - 1}. From a confidence of {k2.prior_confidence} the posterior is {post200.toFixed(3)}. The third notebook's
              notes say a likelihood this small barely moves the posterior; it moves it nearly all the way down.
            </li>
            <li>
              <b>
                <N>κ</N> = {k2.confidence_boost} goes {pct(k2.confidence_boost)} of the way each time.
              </b>{" "}
              So each replay event keeps about {pct(1 - k2.confidence_boost)} of a memory's confidence. At r = 1/{s2.n} every time, getting
              from {k2.prior_confidence} to {k2.floor} takes {events200} replay events, about {Math.round(events200 / s2.replay_prob / 10) * 10}{" "}
              steps. In the figure the highlighted curves reach the floor at about step 70, read off the panel.
            </li>
            <li>
              <b>The slower decay doesn't matter.</b> <N>λ</N>h = {k2.confidence_decay_rate} × {s2.h} = {(k2.confidence_decay_rate * s2.h).toFixed(6)}{" "}
              a step keeps {(keep200 * 100).toFixed(1)}% over {s2.steps.toLocaleString("en")} steps. In test 1 it kept {pct(keep4)}; decay
              wasn't the problem there either.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function Moved2({ num }: { num: number }) {
  const rows = [
    ...bayes200.boosted_top5.map((m) => ({ ...m, group: "Gained most" })),
    ...bayes200.suppressed_top5.map((m) => ({ ...m, group: "Lost most" })),
  ];
  return (
    <Section
      id="moved"
      label="Which moved"
      num={num}
      headline="Softer replay shifts activations by about 1%."
      lead={
        <>
          After about 70 steps every confidence is {k2.floor}, so for almost the whole run this is run C's replay with <N>β</N> ≈{" "}
          {(s2.beta * kFloor200).toFixed(1)} instead of {s2.beta}. The notebook prints the five memories that gained most and the five
          that lost most.
        </>
      }
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Gained and lost most"
            meta="As printed"
            note={`Labels are CIFAR-100 class names; lion appears twice because two of the 200 images are lions. The biggest change, ${maxShift200.toFixed(6)} for the ${cls(maxShiftRow.label)} at α = ${maxShiftRow.decay}, is ${(100 * maxShift200 / maxShiftRow.standard).toFixed(1)}% of its activation.`}
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th className="num">
                      <N>α</N>
                    </th>
                    <th className="num">Standard</th>
                    <th className="num">Bayesian</th>
                    <th className="num">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((m, i) => (
                    <Fragment key={`${m.label}-${m.decay}`}>
                      {(i === 0 || rows[i - 1].group !== m.group) && (
                        <tr key={`h-${m.group}`} className="group-row">
                          <td colSpan={5}>{m.group}</td>
                        </tr>
                      )}
                      <tr key={`${m.label}-${m.decay}`}>
                        <td>{cls(m.label)}</td>
                        <td className="num">{m.decay.toFixed(3)}</td>
                        <td className="num">{f6(m.standard)}</td>
                        <td className="num">{f6(m.bayesian)}</td>
                        <td className={`num big${m.diff > 0 ? " ember-text" : ""}`}>
                          {m.diff > 0 ? "+" : "−"}
                          {Math.abs(m.diff).toFixed(6)}
                        </td>
                      </tr>
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <p className="muted" data-reveal>
            A softer softmax takes replay from memories that get more than their share and gives it to those that get less. So the five
            that lost are memories this run replays more than average, and the five that gained are replayed less. With replay more
            even, decay rate decides more of the order, and ρ rises. Test 3 ranks memories by exactly that share of replay.
          </p>
        </div>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ test 3 */

function Test3View() {
  return (
    <>
      <Change3 num={1} />
      <Result3 num={2} />
      <Saturate3 num={3} />
      <Sets3 num={4} />
      <TwoNumbers3 num={5} />
      <FigureSection
        num={6}
        headline="Confidence splits, step by step."
        label="Figure · Standard against relative evidence, 200 memories"
        src={figures.bayesRelative}
        alt="Six panels. Top: all 200 activations in each run fall from 1 to near 0 within about 500 steps. Middle left: confidence over time; five highlighted curves rise from 0.5 to 1, the first within about 100 steps, five dashed curves fall to 0 by about step 1,000, and many grey curves spread in between. Middle right: final confidence against decay rate, most points at 0 or 1, rho minus 0.027. Bottom left: activation times confidence against decay rate, split into a band near 0.0011 and a band at 0, rho 0.162. Bottom right: Bayesian minus standard activation, within about plus or minus 0.000003."
        note={
          <>
            The notebook's own figure. Top: activations, standard on the left and relative evidence on the right. Middle left: confidence
            over time, the five highest solid and the five lowest dashed. Middle right: final confidence against decay rate. Bottom left:
            activation × confidence against decay rate. Bottom right: the change in each memory's activation against the standard run.
          </>
        }
      />
      <Findings
        num={7}
        shows={[
          <>
            With relative evidence, confidence separates the memories: from 0.0000 to 1.0000, mean {bayesRelative.confidence.mean.toFixed(3)},
            where Bayes' rule with the raw weight sent all {s2.n} to {k2.floor}.
          </>,
          <>
            Confidence has nothing to do with decay rate (ρ = {confRho}), while activation follows it closely
            (ρ = {bayesRelative.bayesian.rho}).
          </>,
          <>The memories it trusts most are the ones test 2's softer replay held back: confidence follows each memory's share of replay.</>,
          <>Activations barely change: the highest and the mean match the standard run to six decimals.</>,
        ]}
        not={[
          <>That confidence measures accuracy or reliability. In this model it counts replay above or below the average, nothing else.</>,
          <>Graded probabilities: with about {events3} standardised updates, any memory with a steady gap ends at 0 or 1.</>,
          <>That the embeddings' similarity structure sets the ranking. It follows from the replay score, but wasn't measured.</>,
          <>
            Other update rates <N>κ</N> or weights <N>w</N>, or other seeds.
          </>,
        ]}
      />
      <CodeSection
        num={8}
        notebook={NOTEBOOKS.bayesRelative}
        path="notebooks/colab/baysian approach/baysian 200 fixed.ipynb"
        output="results/bayesian_200_fixed_output.txt"
        json="results/bayesian_200_fixed.json"
        links={
          <>
            <a className="btn btn-text" href="#/bayes/200">
              <span aria-hidden="true">←</span> Test 2
            </a>
            <a className="btn btn-text" href="#/bayes">
              <span aria-hidden="true">←</span> Test 1
            </a>
          </>
        }
      />
    </>
  );
}

function Change3({ num }: { num: number }) {
  return (
    <Section
      id="test"
      label="The change"
      num={num}
      headline="Count evidence against the average."
      lead={`The third notebook keeps test 2's run and ${s3.n} images and replaces the update. Each memory carries an evidence score e, starting at ${k3.start_evidence}. At every replay event it gains how many standard deviations its replay weight sits above or below the average, times a small rate. Confidence is the sigmoid of e.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="Confidence from evidence" meta="run_bayesian_v3">
            <Tex block>{String.raw`z_i = \frac{r_i - \bar r}{\operatorname{sd}(r)}, \qquad e_i \leftarrow e_i + \kappa\, z_i, \qquad c_i = \sigma(e_i) = \frac{1}{1 + e^{-e_i}}`}</Tex>
            <p className="card-note" style={{ marginTop: 4 }}>
              <N>κ</N> = {k3.conf_lr}, and r̄ = 1/{s3.n} is the mean replay weight. The evidence also decays by <N>λ</N>h e a step with{" "}
              <N>λ</N> = {k3.conf_decay}, which keeps {(keepEvidence * 100).toFixed(1)}% over the run. Replay now weights each activation by{" "}
              {cWeight(k3.conf_weight, 0)} + {k3.conf_weight}c (<N>w</N> = {k3.conf_weight}).
            </p>
          </Card>
          <Card label="Both rules, in log-odds" meta="Worked out from the two updates">
            <Tex block>{String.raw`\operatorname{logit} c_i \;\leftarrow\; \operatorname{logit} c_i + \underbrace{\log \tfrac{r_i}{1 - r_i}}_{\text{tests 1, 2}} \quad\text{or}\quad \operatorname{logit} c_i + \underbrace{\kappa\, z_i}_{\text{test 3}}`}</Tex>
            <p className="card-note" style={{ marginTop: 4 }}>
              In log-odds, Bayes' rule adds the log of each observation's likelihood ratio. Tests 1 and 2 add log(r/(1 − r)), below 0 for
              any memory with less than half of a replay (they also take only part of each step). Test 3's c = <N>σ</N>(e) has the same
              form, a posterior from a prior of 0.5, if <N>κ</N>z is read as one replay's log-likelihood ratio. The difference is that
              the z values of all {s3.n} memories add up to 0: what some memories gain, others lose.
            </p>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function Result3({ num }: { num: number }) {
  const st = bayesRelative.standard;
  const by = bayesRelative.bayesian;
  const cf = bayesRelative.confidence;
  const rows: [string, ReactNode, ReactNode][] = [
    ["Memories left", st.survivors, by.survivors],
    ["Activation ρ, decay rank", st.rho.toFixed(4), by.rho.toFixed(4)],
    ["Confidence ρ, decay rank", "—", neg(bayesRelative.confidence_decay_rho.toFixed(4))],
    ["Confidence spread (highest − lowest)", "—", f4(cf.spread)],
    ["Highest activation", f6(st.max), f6(by.max)],
    ["Mean activation", f6(st.mean), f6(by.mean)],
  ];
  const list = [
    ...bayesRelative.most_confident_top5.map((m) => ({ ...m, group: "Most confident" })),
    ...bayesRelative.least_confident_top5.map((m) => ({ ...m, group: "Least confident" })),
  ];
  return (
    <Section
      id="result"
      label="The result"
      num={num}
      headline="Confidence spreads from 0 to 1."
      lead={`Printed at step ${s3.steps.toLocaleString("en")}: lowest ${f4(cf.min)}, highest ${f4(cf.max)}, mean ${f4(cf.mean)}. All ${s3.n} memories stay above ${s3.alive_threshold}, and activations barely move: the highest and the mean are the standard run's to six decimals.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Confidence" meta="Lowest to highest">
              <div className="stat-value ember">
                {cf.min.toFixed(1)} – {cf.max.toFixed(1)}
              </div>
              <div className="stat-sub">Mean {f4(cf.mean)}, from a start of 0.5 for every memory.</div>
            </Card>
            <Card label="Confidence vs decay" meta="Spearman ρ" delay={1}>
              <div className="stat-value">{confRho}</div>
              <div className="stat-sub">Activation against decay: ρ = {by.rho}.</div>
            </Card>
            <Card label="Memories left" meta={`Above ${s3.alive_threshold}`} delay={2}>
              <div className="stat-value">
                {by.survivors}
                <span className="unit">of {s3.n}</span>
              </div>
              <div className="stat-sub">In both runs.</div>
            </Card>
          </div>
          <Card label="The notebook's summary" meta="As printed">
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th />
                    <th className="num">Standard</th>
                    <th className="num">Relative evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([name, a, b]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td className="num">{a}</td>
                      <td className="num">{b}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card
            label="Most and least confident"
            meta="As printed"
            note="Labels are CIFAR-100 class names; lion and clock each appear twice in the least confident five, as different images of the same class. Confidence prints to four decimals, so 1.0000 and 0.0000 are rounded."
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th className="num">
                      <N>α</N>
                    </th>
                    <th className="num">Activation</th>
                    <th className="num">Confidence</th>
                    <th className="num">a × c</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((m, i) => (
                    <Fragment key={`${m.label}-${m.decay}`}>
                      {(i === 0 || list[i - 1].group !== m.group) && (
                        <tr key={`h-${m.group}`} className="group-row">
                          <td colSpan={5}>{m.group}</td>
                        </tr>
                      )}
                      <tr key={`${m.label}-${m.decay}`}>
                        <td>{cls(m.label)}</td>
                        <td className="num">{m.decay.toFixed(3)}</td>
                        <td className="num">{f6(m.activation)}</td>
                        <td className={`num big${m.confidence > 0.5 ? " ember-text" : ""}`}>{f4(m.confidence)}</td>
                        <td className="num">{f6(m.effective)}</td>
                      </tr>
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function Saturate3({ num }: { num: number }) {
  return (
    <Section
      id="why"
      label="Why 0 and 1"
      num={num}
      headline="Steady evidence piles up until the sigmoid saturates."
      lead={`${s3.steps.toLocaleString("en")} steps at p = ${s3.replay_prob} is about ${events3} replay events. A memory one standard deviation above average at each of them collects e ≈ ${k3.conf_lr} × ${events3} = ${evidence9}, and σ(${evidence9}) = ${f4(sig(evidence9))}; one below ends at ${f4(sig(-evidence9))}.`}
    >
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>z counts rank, not size.</b> At step {s3.steps.toLocaleString("en")} the total activation is about {total3.toFixed(2)} (
              {s3.n} × {f6(bayesRelative.standard.mean)}), so <N>β</N> times any replay score is under {(s3.beta * total3).toFixed(2)}, and
              no memory's replay weight is even {shareRatioBound.toFixed(1)} times another's. Dividing by the spread makes those small
              differences as large as any other: a gap that is small but steady still drives confidence to 0 or 1.
            </li>
            <li>
              <b>Some memories settle fast, some never do.</b> In the figure the highest curve, the maple tree, is at 1 within about 100
              steps; the other highlighted curves reach 0 or 1 by about step 1,500, and many grey curves are still in between at step{" "}
              {s3.steps.toLocaleString("en")}. Read off the panel.
            </li>
            <li>
              <b>So 1.0000 means “replayed more than average, again and again”.</b> It is not a calibrated probability that the memory is
              right.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function Sets3({ num }: { num: number }) {
  const t2Change = (m: { label: string; decay: number }) =>
    [...bayes200.boosted_top5, ...bayes200.suppressed_top5].find((s) => same(s, m))?.diff;
  const list = [
    ...bayesRelative.most_confident_top5.map((m) => ({ ...m, group: "Test 3: most confident" })),
    ...bayesRelative.least_confident_top5.map((m) => ({ ...m, group: "Test 3: least confident" })),
  ];
  return (
    <Section
      id="sets"
      label="What sets it"
      num={num}
      headline="Not decay rate. The share of replay each memory gets."
      lead={`Confidence against decay rate gives ρ = ${confRho}: a slow-decaying memory is no more likely to be trusted. The ${word(topInTest2)} most confident memories are the ${word(topInTest2)} that test 2's softer replay held back most, and ${word(bottomInTest2)} of the five least confident are among the five it helped most.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Test 3 against test 2"
            meta="Matched by class and decay rate"
            note="Test 2 change is Bayesian minus standard activation, as printed. A dash means the memory isn't in test 2's printed five; the notebook doesn't print the rest."
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th className="num">
                      <N>α</N>
                    </th>
                    <th className="num">Test 3 confidence</th>
                    <th className="num">Test 2 change</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((m, i) => {
                    const d = t2Change(m);
                    return (
                      <Fragment key={`${m.label}-${m.decay}`}>
                        {(i === 0 || list[i - 1].group !== m.group) && (
                          <tr key={`h-${m.group}`} className="group-row">
                            <td colSpan={4}>{m.group}</td>
                          </tr>
                        )}
                        <tr key={`${m.label}-${m.decay}`}>
                          <td>{cls(m.label)}</td>
                          <td className="num">{m.decay.toFixed(3)}</td>
                          <td className={`num big${m.confidence > 0.5 ? " ember-text" : ""}`}>{f4(m.confidence)}</td>
                          <td className="num">{d === undefined ? "—" : `${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(6)}`}</td>
                        </tr>
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
          <ul className="points" data-reveal>
            <li>
              <b>Two views of one quantity.</b> Test 2's softer replay takes from memories that get more than the average share of replay;
              test 3 counts that same excess as evidence. The memories one run held back are the ones the other trusts.
            </li>
            <li>
              <b>What decides the share.</b> With normalised replay a memory's score is <Tex>{String.raw`\sum_j \cos(x_i, x_j)\, a_j`}</Tex>.
              Late in the run the activations are close together (mean {f6(bayesRelative.standard.mean)}, highest{" "}
              {f6(bayesRelative.standard.max)}), so the score depends mostly on how similar a memory is to all the others. That points to
              the similarity structure of the embeddings as what sets confidence. The notebook doesn't print each memory's summed
              similarity, so this is a reading of the equation, not a measurement.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function TwoNumbers3({ num }: { num: number }) {
  return (
    <Section
      id="two-numbers"
      label="Two numbers"
      num={num}
      headline="Activation and confidence come apart."
      lead="The notebook multiplies them into an effective strength, a × c. Two images that both decay slowly end with nearly the same activation and opposite confidence."
    >
      <div className="grid">
        <div className="indent stack">
          <div className="grid-2">
            {[trusted, lion].map((m, i) => (
              <Card key={`${m.label}-${m.decay}`} label={cls(m.label)} meta={<><N>α</N> = {m.decay}</>} delay={i}>
                <div className={`stat-value${m.confidence > 0.5 ? " ember" : ""}`}>{f4(m.confidence)}</div>
                <div className="stat-sub">
                  Confidence. Activation {f6(m.activation)}, a × c = {f6(m.effective)}.
                </div>
              </Card>
            ))}
          </div>
          <p className="muted" data-reveal>
            Effective strength against decay rate gives ρ = {bayesRelative.effective_decay_rho_figure} (the figure's bottom-left title),
            against {bayesRelative.bayesian.rho} for activation alone. a × c is computed after the run. Inside it, confidence reaches the
            activations only through the replay weight {cWeight(k3.conf_weight, 0)} + {k3.conf_weight}c, and the figure's bottom-right
            panel puts every memory's change under about 3 × 10⁻⁶.
          </p>
        </div>
      </div>
    </Section>
  );
}
