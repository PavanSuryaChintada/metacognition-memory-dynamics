import { figures, memories, nameOf, NOTEBOOKS, recorded } from "../data/results";
import {
  drive,
  file,
  minDecay,
  run,
  slowest,
  startEnergy,
  startNorm,
  survivor,
  sweepB,
  unnormalisedFinals,
} from "../data/derived";
import { Card, Section } from "../components/Layout";
import { RunIntro } from "../components/RunIntro";
import { ReplaySection } from "../components/ReplaySection";
import { VerdictSection } from "../components/Verdict";
import { DriveChart } from "../components/DriveChart";
import { N } from "../components/ui";

const top = run.retrieval.top[0];
const impliedCosine = top.score / run.final_activations[top.label];

/** Run A: replay scores from the raw embeddings, as the paper writes F(a). Everything from the notebook and replay tracking. */
export function RunUnnormalised() {
  return (
    <>
      <RunIntro
        run="Run A"
        name="Unnormalised replay"
        headline="One memory takes every replay."
        explanation={
          <>
            <p>
              Replay scores every memory against the current state using the raw I-JEPA embeddings, exactly as the paper writes the
              replay operator. Each score is a dot product of 1,280-number vectors, so the scores are large. In the replay-tracking run
              they started near 640 for the top memory and near 625 for the next, read off its figure.
            </p>
            <p>
              Multiplied by β = {recorded.base_params.beta}, gaps that size put all of the softmax's weight on the top score. The{" "}
              {survivor.name.toLowerCase()} photo had the top score at every replay event, took all of the replay, and is the only memory
              left after {recorded.steps} steps. The {nameOf(slowest.label).toLowerCase()}, which decays slowest, went to zero.
            </p>
          </>
        }
        formula={String.raw`F(a) = \mathrm{softmax}\big(\beta\, X^\top X a\big)`}
        formulaNote="X holds the raw embeddings, one per memory. This is the paper's equation for F(a), unchanged."
        finals={unnormalisedFinals}
        outcomeLabel={`Figure 1 · Activation after ${recorded.steps} steps`}
        outcomeMeta={`The notebook run · seed ${recorded.seed} · η = ${run.eta}, p = ${run.replay_prob}`}
      />
      <SingleRunSection num={1} />
      <ReplaySection num={2} />
      <RetrievalSection num={3} />
      <RegimesSection num={4} />
      <VerdictSection num={5} startEnergy={startEnergy} minDecay={minDecay} />
      <RunCodeSection num={6} />
    </>
  );
}

function SingleRunSection({ num }: { num: number }) {
  return (
    <Section
      id="single-run"
      label="One run"
      num={num}
      headline="One memory survives."
      lead={`Replay strength η = ${run.eta}, replay probability p = ${run.replay_prob}, ${recorded.steps} steps, seed ${recorded.seed}. Three memories reach exactly 0.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Survivors" meta={`Step ${recorded.steps}`}>
              <div className="stat-value ember">
                1<span className="unit">of {recorded.memories.length}</span>
              </div>
              <div className="squares" aria-hidden="true">
                {memories.map((m) => (
                  <span key={m.label} className={m.final > 0 ? "alive" : undefined} title={m.name} />
                ))}
              </div>
              <div className="stat-sub">
                The {survivor.name.toLowerCase()} photo keeps {survivor.final.toFixed(4)}.
              </div>
            </Card>
            <Card label="Energy" meta="E(a), start to step 500" delay={1}>
              <div className="stat-value">
                {startEnergy.toFixed(2)} <span className="unit">to</span> {run.final_energy.toFixed(4)}
              </div>
              <div className="stat-sub">About 200 times lower, then flat where replay adds back what decay removes.</div>
            </Card>
            <Card label="Slowest decay" meta={nameOf(slowest.label)} delay={2}>
              <div className="stat-value">
                <N>α</N> = {minDecay.toFixed(2)}
              </div>
              <div className="stat-sub">The slowest-fading memory, and it still went to 0.</div>
            </Card>
          </div>
          <Card label="Decay rate against outcome" meta="Lower α means slower decay" title="Replay picked the winner, not decay.">
            <table className="outcome-table">
              <thead>
                <tr>
                  <th>Memory</th>
                  <th className="num">
                    Decay rate <N>α</N>
                  </th>
                  <th className="num">Activation at step {recorded.steps}</th>
                </tr>
              </thead>
              <tbody>
                {[...memories]
                  .sort((a, b) => a.decay_rate - b.decay_rate)
                  .map((m) => (
                    <tr key={m.label} className={m.final > 0 ? "win" : undefined}>
                      <td>
                        {m.name}
                        {m.label === slowest.label && <span className="tag">slowest decay</span>}
                        {m.final > 0 && <span className="tag ember">survived</span>}
                      </td>
                      <td className="num big">{m.decay_rate.toFixed(2)}</td>
                      <td className="num big">{m.final > 0 ? m.final.toFixed(4) : "0"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function RetrievalSection({ num }: { num: number }) {
  return (
    <Section
      id="retrieval"
      label="Retrieval"
      num={num}
      headline="The exact match ranks last."
      lead="After the run the memory is queried with the beagle photo, the same image as memory 1. Score is cosine similarity times current activation."
    >
      <div className="grid">
        <div className="indent-narrow stack">
          <Card label="Top retrievals" meta="Query: the beagle photo">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Memory</th>
                  <th className="num">Score</th>
                </tr>
              </thead>
              <tbody>
                {run.retrieval.top.map((t, i) => (
                  <tr key={t.label}>
                    <td>{i + 1}</td>
                    <td>{nameOf(t.label)}</td>
                    <td className="num big">{t.score.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <p className="muted" data-reveal>
            The beagle's cosine similarity to the query is 1, but its activation is 0, so it scores 0. Once activations collapse, only
            the surviving memory can score above zero, whatever the query. Ranks 2 and 3 are tied at 0, so their order means nothing.
            The top score also gives one similarity for free: {top.score.toFixed(4)} / {run.final_activations[top.label].toFixed(4)} ≈{" "}
            {impliedCosine.toFixed(2)} for beagle and person. The replay-tracking run measured it directly: 0.3005.
          </p>
        </div>
      </div>
    </Section>
  );
}

function RegimesSection({ num }: { num: number }) {
  return (
    <Section
      id="regimes"
      label="Regimes"
      num={num}
      headline="Three regimes, ordered by replay."
      lead="Section 4 of the paper predicts collapse, a balanced state and runaway reinforcement as replay gets stronger. The same four memories were run under three settings of η and p."
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label={
              <>
                Figure 3 · Final <N>‖a‖</N> against replay drive
              </>
            }
            meta="Log scales · seed 42 · 500 steps"
            numbers={
              <table>
                <thead>
                  <tr>
                    <th>Regime</th>
                    <th className="num">
                      <N>η</N>
                    </th>
                    <th className="num">
                      <N>p</N>
                    </th>
                    <th className="num">
                      <N>η·p</N>
                    </th>
                    <th className="num">
                      Final <N>‖a‖</N>
                    </th>
                    <th className="num">Final energy</th>
                  </tr>
                </thead>
                <tbody>
                  {sweepB.map((r) => (
                    <tr key={r.name}>
                      <td>{r.name}</td>
                      <td className="num">{r.eta}</td>
                      <td className="num">{r.replay_prob}</td>
                      <td className="num">{drive(r)}</td>
                      <td className="num">{r.final_norm}</td>
                      <td className="num">{r.final_energy}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <DriveChart
              exact={sweepB.map((r) => ({ name: r.name, drive: drive(r), norm: r.final_norm }))}
              approx={[]}
              start={startNorm}
            />
          </Card>
          <Card
            label="Figure 4 · The three regimes"
            meta="Colab output"
            note={`The printed values come from a second run of each setting with fresh random draws, so the Balanced curve here ends near 0.47 while the table says ${sweepB[1].final_norm}.`}
          >
            <div className="plate">
              <img
                src={figures.sweepB}
                alt="Colab figure: Collapse decays smoothly to zero; Balanced drops to around 0.5 and hovers; Runaway climbs from 2 to about 6 and levels off."
              />
            </div>
          </Card>
          <ul className="points" data-reveal>
            <li>
              <b>Collapse</b>, η·p = {drive(sweepB[0])}. Replay is too weak to matter; everything decays to ‖a‖ = {sweepB[0].final_norm}.
            </li>
            <li>
              <b>Balanced</b>, η·p = {drive(sweepB[1])}. ‖a‖ drops over the first 100 steps, then hovers between about 0.45 and 0.65.
            </li>
            <li>
              <b>Runaway</b>, η·p = {drive(sweepB[2])}. ‖a‖ climbs from 2 to about 6 and levels off; energy ends at{" "}
              {sweepB[2].final_energy.toFixed(1)}, about {Math.round(sweepB[2].final_energy / startEnergy)} times where it started.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function RunCodeSection({ num }: { num: number }) {
  return (
    <Section id="run-code" label="The code" num={num} headline="Where these numbers come from.">
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>The notebook run.</b> <a href={NOTEBOOKS.original}>notebooks/metacognition.ipynb</a>, printed output in{" "}
              <a href={file("results/notebook_output.txt")}>results/notebook_output.txt</a>, numbers in{" "}
              <a href={file("results/metrics.json")}>results/metrics.json</a>.
            </li>
            <li>
              <b>Replay tracking.</b> <a href={NOTEBOOKS.unnormalised4}>notebooks/colab/unnormalised/metacognition_4_memories.ipynb</a>,
              the cell headed TRUE RUN 1 REPRODUCTION WITH F(a) TRACKING. Printed output in{" "}
              <a href={file("results/replay_tracking_output.txt")}>results/replay_tracking_output.txt</a>, numbers in{" "}
              <a href={file("results/replay_tracking.json")}>results/replay_tracking.json</a>. The exploratory cells before it use other
              settings and aren't used on this site.
            </li>
          </ul>
          <a className="btn btn-primary" href="#/normalised" style={{ alignSelf: "flex-start" }}>
            Next: run B, normalised replay
          </a>
        </div>
      </div>
    </Section>
  );
}
