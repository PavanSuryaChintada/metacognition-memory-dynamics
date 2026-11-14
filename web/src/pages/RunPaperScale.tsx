import { figures, NOTEBOOKS, paperScale, paperTuned } from "../data/results";
import { file } from "../data/derived";
import { Card, Section } from "../components/Layout";
import { RunIntro } from "../components/RunIntro";
import { ThresholdMeter } from "../components/RunCards";
import { Tex } from "../components/Tex";
import { N } from "../components/ui";
import { SurvivorGrid } from "../components/SurvivorGrid";
import { PaperTunedView } from "./PaperTuned";

const ps = paperScale.setup;
const cos = paperScale.cosine_off_diagonal;
// decay rates are evenly spaced from 0.5 to 1.5, so their mean is 1
const meanDecay = (0.5 + 1.5) / 2;
const inhibitionPerUnit = ps.delta * ps.n; // δN: what inhibition takes per unit of total activation
const longRunTotal = (ps.eta * ps.replay_prob) / (meanDecay + inhibitionPerUnit);
const longRunEach = longRunTotal / ps.n;
const ceilingEach = ps.eta / (meanDecay + inhibitionPerUnit) / ps.n; // replay at every single step
const neededTotal = ps.alive_threshold * ps.n;
const shrinkPerStep = ps.h * (meanDecay + inhibitionPerUnit);
const jumpPerReplay = ps.h * ps.eta;
// steps for every memory to fall below the line from 1 with no replay, all memories equal
const stepsToFall = (() => {
  let a = 1;
  let t = 0;
  while (a > ps.alive_threshold) {
    a *= 1 - shrinkPerStep;
    t += 1;
  }
  return t;
})();
const totalAt3000 = paperScale.mean_activation * ps.n;
const shareSpread = Math.exp(ps.beta * totalAt3000) - 1;
const lineShare = paperScale.max_activation / ps.alive_threshold;

/** Run C: the paper's N and T with 200 real images and normalised replay, in two tests. */
export function RunPaperScale({ view }: { view: 1 | 2 }) {
  return (
    <>
      <RunIntro
        run="Run C"
        name="Paper scale"
        headline="At the paper's scale, inhibition sets the level."
        explanation={
          <>
            <p>
              The paper's experiments use {ps.n} memories and {ps.steps.toLocaleString("en")} steps. This run uses the same N and T with
              real images: the first {ps.n} images of the CIFAR-100 test set, each 32 × 32 pixels, enlarged to 224 × 224 and encoded with
              I-JEPA. Replay is normalised, as in run B.
            </p>
            <p>
              Inhibition takes <N>δ</N> times the total activation off every memory, so with {ps.n} memories the total loses{" "}
              <N>δ</N>N times itself per unit of time. Test 1 uses the paper's <N>δ</N> = {ps.delta}: that is{" "}
              {Math.round(inhibitionPerUnit)}, against decay of 0.5 to 1.5, and none of the {ps.n} stays above {ps.alive_threshold}. Test 2
              sweeps <N>γ</N> and <N>δ</N> to find where they do. Pick one.
            </p>
          </>
        }
        formula={String.raw`F(a) = \mathrm{softmax}\big(\beta\, \hat X^\top \hat X a\big)`}
        formulaNote="The same normalised replay as run B. What changes is the scale: 200 memories for 3,000 steps."
      />
      <PaperTestCards view={view} />
      {view === 1 ? <Test1View /> : <PaperTunedView />}
    </>
  );
}

const lowDelta = Math.min(...paperTuned.setup.deltas);
const bestTuned = paperTuned.best_printed;

function PaperTestCards({ view }: { view: 1 | 2 }) {
  return (
    <section className="section" id="paper-tests">
      <div className="wrap">
        <div className="run-cards beyond-cards">
          <a className={`run-card experiment-card${view === 1 ? " on" : ""}`} href="#/paper" aria-current={view === 1 ? "page" : undefined}>
            <div className="run-card-head">
              <span className="label">Test 1</span>
              {view === 1 && <span className="tag new">Showing</span>}
            </div>
            <h2 className="run-title">The paper's γ and δ</h2>
            <p className="run-desc">
              γ = {ps.gamma} and δ = {ps.delta}, as in the paper. Every memory collapses within the first few dozen steps and none stays
              above {ps.alive_threshold}.
            </p>
            <ThresholdMeter />
            <dl className="run-stats">
              <div>
                <dt>Memories left</dt>
                <dd>
                  {paperScale.survivors} of {ps.n}
                </dd>
              </div>
              <div>
                <dt>Order by decay</dt>
                <dd>ρ = {paperScale.spearman_rho_decay_vs_activation.toFixed(3)}</dd>
              </div>
            </dl>
          </a>
          <a className={`run-card experiment-card${view === 2 ? " on" : ""}`} href="#/paper/tuned" aria-current={view === 2 ? "page" : undefined}>
            <div className="run-card-head">
              <span className="label">Test 2</span>
              {view === 2 && <span className="tag new">Showing</span>}
            </div>
            <h2 className="run-title">Sweeping γ and δ</h2>
            <p className="run-desc">
              25 runs over five values of each. Only δ = {lowDelta} keeps memories above the line, and their average sits right on it.
            </p>
            <SurvivorGrid />
            <dl className="run-stats">
              <div>
                <dt>Memories left</dt>
                <dd>
                  {bestTuned.alive} of {ps.n} <span className="dd-sub">δ = {lowDelta}, γ = 0.2 or {bestTuned.gamma}</span>
                </dd>
              </div>
              <div>
                <dt>Lowest memory</dt>
                <dd>
                  {bestTuned.min_alive.toFixed(6)} <span className="dd-sub">the line is {ps.alive_threshold}</span>
                </dd>
              </div>
            </dl>
          </a>
        </div>
      </div>
    </section>
  );
}

function Test1View() {
  return (
    <>
      <SetupSection num={1} />
      <ResultSection num={2} />
      <WhySection num={3} />
      <ReplaySpreadSection num={4} />
      <OrderSection num={5} />
      <FigureSection num={6} />
      <FindingsSection num={7} />
      <CodeSection num={8} />
    </>
  );
}

function SetupSection({ num }: { num: number }) {
  return (
    <Section
      id="setup"
      label="The setup"
      num={num}
      headline="The paper's N and T, with real images."
      lead="Everything that differs from the paper's own experiments, and the similarity of the embeddings that came out."
    >
      <div className="grid">
        <div className="indent stack">
          <div className="card" data-reveal>
            <table className="compare">
              <thead>
                <tr>
                  <th />
                  <th>Paper</th>
                  <th>Run C</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Memories N</td>
                  <td data-label="Paper">200</td>
                  <td data-label="Run C">{ps.n}</td>
                </tr>
                <tr>
                  <td>Steps</td>
                  <td data-label="Paper">3,000, averaged over 10 seeds</td>
                  <td data-label="Run C">
                    {ps.steps.toLocaleString("en")}, one seed ({ps.seed})
                  </td>
                </tr>
                <tr>
                  <td>Embeddings</td>
                  <td data-label="Paper">Random draws from a standard Gaussian, d = 64</td>
                  <td data-label="Run C">I-JEPA features of CIFAR-100 test images (32 × 32, enlarged to 224 × 224), d = 1,280</td>
                </tr>
                <tr>
                  <td>Cosine similarity between memories</td>
                  <td data-label="Paper">Centres on 0, spread about 1/√64 ≈ 0.13</td>
                  <td data-label="Run C">
                    {cos.min.toFixed(2)} to {cos.max.toFixed(2)}, mean {cos.mean.toFixed(2)}, spread {cos.std.toFixed(2)}
                  </td>
                </tr>
                <tr>
                  <td>Replay scores</td>
                  <td data-label="Paper">Raw embeddings</td>
                  <td data-label="Run C">Unit-length embeddings, as in run B</td>
                </tr>
                <tr>
                  <td>Decay rates</td>
                  <td data-label="Paper">α ~ U[0.5, 1.5]</td>
                  <td data-label="Run C">Evenly spaced from 0.5 to 1.5</td>
                </tr>
                <tr>
                  <td>Parameters</td>
                  <td colSpan={2}>
                    Run C: γ = {ps.gamma}, δ = {ps.delta}, β = {ps.beta}, η = {ps.eta}, p = {ps.replay_prob}, h = {ps.h}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="muted" data-reveal>
            For comparison, the four photos in runs A and B have cosine similarities between 0.30 and 0.47. Embedding lengths here run from{" "}
            {paperScale.embedding_norms.min} to {paperScale.embedding_norms.max}.
          </p>
        </div>
      </div>
    </Section>
  );
}

function ResultSection({ num }: { num: number }) {
  return (
    <Section
      id="result"
      label="The result"
      num={num}
      headline={`${paperScale.survivors} of ${ps.n} above ${ps.alive_threshold}.`}
      lead={`The highest activation at step ${ps.steps.toLocaleString("en")} is ${paperScale.max_activation.toFixed(6)}, ${Math.round(lineShare * 100)}% of the ${ps.alive_threshold} line. The mean of all ${ps.n} is ${paperScale.mean_activation.toFixed(6)}.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Memories left" meta={`Above ${ps.alive_threshold}`}>
              <div className="stat-value ember">
                {paperScale.survivors}
                <span className="unit">of {ps.n}</span>
              </div>
              <div className="stat-sub">At every printed checkpoint, from step 500 on.</div>
            </Card>
            <Card label="Highest at the end" meta={`Step ${ps.steps.toLocaleString("en")}`} delay={1}>
              <div className="stat-value">{paperScale.max_activation.toFixed(6)}</div>
              <ThresholdMeter />
            </Card>
            <Card label="Order by decay rate" meta="Spearman ρ" delay={2}>
              <div className="stat-value">{paperScale.spearman_rho_decay_vs_activation.toFixed(3)}</div>
              <div className="stat-sub">Lower decay, higher activation, by tiny margins. See (05).</div>
            </Card>
          </div>
          <Card label="Printed checkpoints" meta="Rounded to 4 decimals by the notebook">
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th className="num">Step</th>
                    <th className="num">Above {ps.alive_threshold}</th>
                    <th className="num">Highest</th>
                    <th className="num">Mean</th>
                  </tr>
                </thead>
                <tbody>
                  {paperScale.progress.map((r) => (
                    <tr key={r.step}>
                      <td className="num">{r.step.toLocaleString("en")}</td>
                      <td className="num">
                        {r.alive} of {ps.n}
                      </td>
                      <td className="num">{r.max}</td>
                      <td className="num">{r.mean}</td>
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

function WhySection({ num }: { num: number }) {
  return (
    <Section
      id="why"
      label="Why"
      num={num}
      headline="Inhibition grows with the number of memories."
      lead="Add the paper's equation up over every memory. Diffusion drops out, because the similarity graph is symmetric and what one memory gives another receives. The replay weights add up to 1. What is left is an equation for the total."
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="The total activation" meta="From the paper's equation, summed">
            <Tex block>{String.raw`\frac{d}{dt}\sum_i a_i \;=\; -\sum_i \alpha_i a_i \;-\; \delta N \sum_i a_i \;+\; \eta\, S(t)`}</Tex>
            <p className="card-note" style={{ marginTop: 4 }}>
              The embeddings don't appear in it. It holds while no memory is held at 0 by the clamp; in the figure every memory ends a little above 0.
            </p>
          </Card>
          <ul className="points" data-reveal>
            <li>
              <b>Inhibition is {Math.round(inhibitionPerUnit)} times the total.</b> With <N>δ</N> = {ps.delta} and N = {ps.n}, inhibition
              takes {Math.round(inhibitionPerUnit)} times the total away per unit of time, against decay of 0.5 to 1.5. At the start, with every memory at 1,
              the total shrinks by about {Math.round(shrinkPerStep * 100)}% a step; every memory is below {ps.alive_threshold} within about{" "}
              {stepsToFall} steps. The figure's curves drop to near 0 right at the start.
            </li>
            <li>
              <b>Replay can't refill it.</b> Replay adds <N>η</N> = {ps.eta} to the total when it fires, on {Math.round(ps.replay_prob * 100)}% of steps.
              In the long run the total settles around <N>η</N>p / (mean <N>α</N> + <N>δ</N>N) = {(ps.eta * ps.replay_prob).toFixed(2)} /{" "}
              {Math.round(meanDecay + inhibitionPerUnit)} ≈ {longRunTotal.toFixed(4)}, about {longRunEach.toFixed(6)} per memory. Even if replay fired
              every step, the average couldn't stay above about {ceilingEach.toFixed(5)}.
            </li>
            <li>
              <b>The line needs {(neededTotal / longRunTotal).toFixed(1)} times more.</b> All {ps.n} above{" "}
              {ps.alive_threshold} needs a total of at least {neededTotal}; the equation allows about {longRunTotal.toFixed(3)}.
            </li>
            <li>
              <b>The printed mean is one moment.</b> When replay fires the total jumps by h<N>η</N> = {jumpPerReplay.toFixed(3)}, then
              shrinks by about a fifth each step, so a single step's mean moves around the long-run value. At step{" "}
              {ps.steps.toLocaleString("en")} it is {paperScale.mean_activation.toFixed(6)}, a total of {totalAt3000.toFixed(4)}.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function ReplaySpreadSection({ num }: { num: number }) {
  return (
    <Section
      id="replay"
      label="Replay"
      num={num}
      headline="With activations this small, replay is spread evenly."
      lead="With unit-length embeddings and every cosine similarity above 0, each replay score sits between 0 and the total activation. So when the total is tiny, all the scores are close together and the softmax gives every memory almost the same share."
    >
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              At step {ps.steps.toLocaleString("en")} the total is {totalAt3000.toFixed(4)}, so <N>β</N> times any score is at most{" "}
              {(ps.beta * totalAt3000).toFixed(3)}, and no memory's replay share is more than {Math.round(shareSpread * 100)}% above
              another's: close to 1/{ps.n} each.
            </li>
            <li>
              The figure's bottom-right panel agrees: replay entropy sits at its maximum, ln {ps.n} ≈ {Math.log(ps.n).toFixed(2)}, which is
              what an even split gives. The notebook doesn't print the entropy.
            </li>
            <li>
              So the embeddings' similarity, mean {cos.mean.toFixed(2)} here, isn't what holds the memories down. It isn't in the equation
              for the total, and with scores this small it barely changes how replay is shared.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function OrderSection({ num }: { num: number }) {
  return (
    <Section
      id="order"
      label="Decay order"
      num={num}
      headline="Ordered by decay rate, by margins too small to print."
      lead={`Spearman ρ between decay rate and final activation is ${paperScale.spearman_rho_decay_vs_activation} (p = ${paperScale.spearman_p}): the lower the decay rate, the higher the activation. But the top ten are equal to six decimals.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Top ten at the end"
            meta="As printed"
            note={`Labels are CIFAR-100 class names; two of the ten are images of the class sea. The notebook's "Bottom 10, still alive" list printed nothing, since none are above ${ps.alive_threshold}.`}
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th className="num">Rank</th>
                    <th>Class</th>
                    <th className="num">
                      Decay rate <N>α</N>
                    </th>
                    <th className="num">Activation</th>
                  </tr>
                </thead>
                <tbody>
                  {paperScale.top10.map((r) => (
                    <tr key={r.rank}>
                      <td className="num">{r.rank}</td>
                      <td>{r.label}</td>
                      <td className="num">{r.decay.toFixed(4)}</td>
                      <td className="num big">{r.activation.toFixed(6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <p className="muted" data-reveal>
            The notebook ends with “Does the balanced regime hold with real embeddings? YES”. That line is printed whenever ρ is above 0.5;
            it doesn't look at how many memories are left, and none are.
          </p>
        </div>
      </div>
    </Section>
  );
}

function FigureSection({ num }: { num: number }) {
  return (
    <Section id="figure" label="Figure" num={num} headline="All 200, step by step.">
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Figure · Paper scale on real I-JEPA"
            meta="Colab output"
            note={
              <>
                The notebook's own figure. Top: all {ps.n} activations, the top ten highlighted, and final activation against decay rate
                with the {ps.alive_threshold} line. Bottom: survival by decay-rate bin and the distribution of survivors, both empty since
                none survive, and replay entropy over time against its maximum.
              </>
            }
          >
            <div className="plate">
              <img
                src={figures.paperScale}
                alt="Six panels. All 200 activations drop from 1 to near 0 within the first few dozen steps and stay there. Final activation against decay rate is a flat line just under 0.0001, far below the 0.001 line. Survival bins and the survivor histogram are empty. Replay entropy sits at its maximum, about 5.3."
              />
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function FindingsSection({ num }: { num: number }) {
  return (
    <Section id="findings" label="Findings" num={num} headline="What it shows, and what it doesn't.">
      <div className="grid">
        <div className="indent two-col">
          <div data-reveal>
            <div className="label">Shows</div>
            <ul className="points">
              <li>
                With {ps.n} memories, <N>δ</N> = {ps.delta}, <N>η</N> = {ps.eta} and p = {ps.replay_prob}, normalised replay leaves none
                above {ps.alive_threshold}.
              </li>
              <li>
                The paper's equation explains it on its own: summed over memories, inhibition takes {Math.round(inhibitionPerUnit)} times
                the total per unit of time, and replay can hold the total near {longRunTotal.toFixed(3)}, far below the {neededTotal} that {ps.n} memories at{" "}
                {ps.alive_threshold} need.
              </li>
              <li>The final activations still follow decay rate, by margins under 0.000001.</li>
            </ul>
          </div>
          <div data-reveal>
            <div className="label">Doesn't show</div>
            <ul className="points">
              <li>That the embeddings' high similarity causes it: the total doesn't depend on the embeddings.</li>
              <li>What the paper's own setup, raw Gaussian embeddings, gives with these settings: not run here.</li>
              <li>Whether a larger <N>η</N> keeps memories alive: not run. Lower <N>δ</N> is test 2.</li>
              <li>Anything about other images or seeds: one set of 200 CIFAR-100 images, one seed.</li>
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
}

function CodeSection({ num }: { num: number }) {
  return (
    <Section id="run-code" label="The code" num={num} headline="Where these numbers come from.">
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>The executed notebook.</b> <a href={NOTEBOOKS.paperScale}>notebooks/colab/paper/paper_scale_real_jepa.ipynb</a>, with its
              code, printed output and figure as they ran in Colab.
            </li>
            <li>
              <b>Recorded copies.</b> Printed output in <a href={file("results/paper_scale_output.txt")}>results/paper_scale_output.txt</a>,
              numbers in <a href={file("results/paper_scale.json")}>results/paper_scale.json</a>.
            </li>
          </ul>
          <div className="btn-row">
            <a className="btn btn-primary" href={NOTEBOOKS.paperScale}>
              Open the Colab notebook on GitHub
            </a>
            <a className="btn btn-text" href="#/paper/tuned">
              Next: test 2, sweeping γ and δ <span aria-hidden="true">→</span>
            </a>
          </div>
        </div>
      </div>
    </Section>
  );
}
