import { figures, memories, nameOf, normalised, normalisedStartShares, recorded, replay } from "../data/results";
import { file, lateShareRange, normalisedFinals } from "../data/derived";
import { Card, Section } from "../components/Layout";
import { RunIntro } from "../components/RunIntro";
import { N } from "../components/ui";

const raw = normalised.final_activations.unnormalised;
const norm = normalised.final_activations.normalised;
const late = normalised.replay_weight_mean_last_10pct;
const beta = normalised.setup.beta;
const startShares = normalisedStartShares(beta);
const finalsSum = Object.values(norm).reduce((a, b) => a + b, 0);
const byDecay = [...memories].sort((a, b) => a.decay_rate - b.decay_rate);
const top = byDecay[0];
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const sim = replay.cosine_similarity.flatMap((row, i) => row.filter((_, j) => j !== i));

/** Run B: the same run with replay scores from embeddings scaled to length 1. From the third Colab run. */
export function RunNormalised() {
  return (
    <>
      <RunIntro
        run="Run B"
        name="Normalised replay"
        headline="Replay spreads, and all four memories stay."
        explanation={
          <>
            <p>
              The only change is inside the replay operator: each embedding is scaled to length 1 before the replay scores are computed,
              so a memory's score is the other memories' activations weighted by cosine similarity. Decay, diffusion, inhibition, the
              embeddings, the decay rates and the replay coin flips are the same as in the unnormalised arm of the same script.
            </p>
            <p>
              Cosine similarities between these photos run from {Math.min(...sim).toFixed(2)} to {Math.max(...sim).toFixed(2)}, so the
              scores stay small and close together. The softmax spreads replay over all four memories, all four are still active after{" "}
              {normalised.setup.steps} steps, and their final activations line up with their decay rates: the slowest-decaying{" "}
              {top.name.toLowerCase()} ends highest and the fastest-decaying {byDecay[byDecay.length - 1].name.toLowerCase()} lowest.
            </p>
          </>
        }
        formula={String.raw`F(a) = \mathrm{softmax}\big(\beta\, \hat X^\top \hat X a\big)`}
        formulaNote="X̂ is X with every embedding scaled to length 1, so X̂ᵀX̂ holds cosine similarities. Only F(a) uses it; diffusion already used cosine similarity in both runs."
        finals={normalisedFinals}
        outcomeLabel={`Figure 1 · Activation after ${normalised.setup.steps} steps`}
        outcomeMeta={`Normalised arm · seed ${normalised.setup.seed} · η = ${normalised.setup.eta}, p = ${normalised.setup.replay_prob}`}
      />
      <ChangeSection num={1} />
      <ResultSection num={2} />
      <SpreadSection num={3} />
      <FigureSection num={4} />
      <FindingsSection num={5} />
      <CodeSection num={6} />
    </>
  );
}

function ChangeSection({ num }: { num: number }) {
  return (
    <Section
      id="change"
      label="The change"
      num={num}
      headline="Two lines in the replay operator."
      lead="The comparison script runs both versions back to back on the same encoded photos. This is everything that differs between them."
    >
      <div className="grid">
        <div className="indent stack">
          <div className="grid-2">
            <Card label="Run A · raw" meta="As the paper writes F(a)">
              <pre className="code-light">
                <code>{"s = X @ (X.T @ a)\nF = softmax(beta * s)"}</code>
              </pre>
            </Card>
            <Card label="Run B · normalised" meta="Each row of X scaled to length 1" delay={1}>
              <pre className="code-light">
                <code>{"X_n = normalize(X, dim=1)\ns = X_n @ (X_n.T @ a)\nF = softmax(beta * s)"}</code>
              </pre>
            </Card>
          </div>
          <Card label="The same in both arms" meta="normalised_replay.py">
            <ul className="points compact">
              <li>The four I-JEPA embeddings, encoded once.</li>
              <li>Decay rates {normalised.decay_rates_printed.map((r) => r.toFixed(2)).join(", ")}.</li>
              <li>
                <N>η</N> = {normalised.setup.eta}, <N>β</N> = {beta}, <N>γ</N> = {normalised.setup.gamma}, <N>δ</N> ={" "}
                {normalised.setup.delta}, h = {normalised.setup.h}, p = {normalised.setup.replay_prob}, {normalised.setup.steps} steps.
              </li>
              <li>
                The replay coin flips: the seed is reset before each arm, so replay fired at the same {normalised.replay_events.normalised}{" "}
                steps in both.
              </li>
            </ul>
          </Card>
          <p className="muted" data-reveal>
            The paper writes the replay operator with the raw embeddings, so run B is a change to the model, not a correction of the code.
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
      headline="All four memories stay, in decay order."
      lead="Final activations and replay shares for both arms of the comparison script."
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Memories left" meta="Above 0.001 at step 500">
              <div className="stat-value ember">
                {normalised.survivors_above_0_001.normalised}
                <span className="unit">of 4</span>
              </div>
              <div className="stat-sub">The raw arm keeps {normalised.survivors_above_0_001.unnormalised} of 4.</div>
            </Card>
            <Card label="Replay share each" meta="Last 10% of replay events" delay={1}>
              <div className="stat-value">
                {pct(lateShareRange[0])} <span className="unit">to</span> {pct(lateShareRange[1])}
              </div>
              <div className="stat-sub">The raw arm gives 100% to the person.</div>
            </Card>
            <Card label="Highest at the end" meta={nameOf(top.label)} delay={2}>
              <div className="stat-value">{norm[top.label].toFixed(4)}</div>
              <div className="stat-sub">
                The memory with the slowest decay, <N>α</N> = {top.decay_rate.toFixed(2)}.
              </div>
            </Card>
          </div>
          <Card
            label="Activation at step 500"
            meta="Sorted by decay rate"
            note={`In this script the seed is reset after the decay rates are drawn, so the coin flips start four draws earlier than in the notebook. That is why the raw arm ends at ${raw.person} here and at ${recorded.single_run.final_activations.person} in the notebook.`}
          >
            <table>
              <thead>
                <tr>
                  <th>Memory</th>
                  <th className="num">
                    Decay rate <N>α</N>
                  </th>
                  <th className="num">Raw arm</th>
                  <th className="num">Normalised arm</th>
                </tr>
              </thead>
              <tbody>
                {byDecay.map((m) => (
                  <tr key={m.label}>
                    <td>{m.name}</td>
                    <td className="num">{m.decay_rate.toFixed(2)}</td>
                    <td className="num big">{raw[m.label] > 0 ? raw[m.label].toFixed(6) : "0"}</td>
                    <td className="num big ember-text">{norm[m.label].toFixed(6)}</td>
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

function SpreadSection({ num }: { num: number }) {
  return (
    <Section
      id="spread"
      label="Replay weights"
      num={num}
      headline="Uneven at the start, even by the end."
      lead="With unit-length embeddings, a replay score is a sum of cosine similarities weighted by activation. That keeps the scores small, so the softmax never concentrates."
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Replay share by memory"
            meta="Normalised arm"
            note={
              <>
                Start: worked out from the measured cosine similarities with every activation at 1, using the script's own formula. The
                figure's first replay weights show the same split. End: the recorded mean over the last 10% of replay events.
              </>
            }
          >
            <table>
              <thead>
                <tr>
                  <th>Memory</th>
                  <th className="num">At the start (computed)</th>
                  <th className="num">Last 10% (recorded)</th>
                </tr>
              </thead>
              <tbody>
                {replay.labels.map((l, i) => (
                  <tr key={l}>
                    <td>{nameOf(l)}</td>
                    <td className="num">{pct(startShares[i])}</td>
                    <td className="num big">{pct(late.normalised[l])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <p className="muted" data-reveal>
            Why it evens out: each score is at most the sum of all activations. At step {normalised.setup.steps} that sum is{" "}
            {finalsSum.toFixed(3)}, so <N>β</N> times any score is at most {(beta * finalsSum).toFixed(2)}, and no memory's share can be
            more than {Math.exp(beta * finalsSum).toFixed(1)} times another's. The recorded shares sit between {pct(lateShareRange[0])}{" "}
            and {pct(lateShareRange[1])}.
          </p>
        </div>
      </div>
    </Section>
  );
}

function FigureSection({ num }: { num: number }) {
  return (
    <Section id="figure" label="Figure" num={num} headline="Both arms, step by step.">
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Figure 2 · Raw and normalised replay"
            meta="Colab output"
            note={
              <>
                The run's own figure. Top row: raw arm. Bottom row: normalised arm. Its labels are from that run: <code>labrador</code> is
                the beagle and <code>cute_dog</code> the French bulldog.
              </>
            }
          >
            <div className="plate">
              <img
                src={figures.normalised}
                alt="Eight panels. Top row, raw replay: only the person stays above zero and takes all replay weight. Bottom row, normalised replay: all four activations settle between about 0.03 and 0.05 and the replay weights converge to about 0.25 each."
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
              <li>Scaling each embedding to length 1 inside F(a) changes the outcome from one memory left to all four.</li>
              <li>
                Replay weight spreads to between {pct(lateShareRange[0])} and {pct(lateShareRange[1])} per memory by the end.
              </li>
              <li>
                The final activations follow the decay rates:{" "}
                {byDecay.map((m) => `${m.name.toLowerCase()} ${norm[m.label].toFixed(4)}`).join(", ")}.
              </li>
            </ul>
          </div>
          <div data-reveal>
            <div className="label">Doesn't show</div>
            <ul className="points">
              <li>
                How the collapse, balanced and runaway regimes look with normalised scores. Only η = {normalised.setup.eta}, p ={" "}
                {normalised.setup.replay_prob} was run.
              </li>
              <li>Anything statistical: one seed and four memories.</li>
              <li>Whether a setting between the two gives replay that favours some memories without taking all of it.</li>
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
              <b>The comparison script.</b> <a href={file("notebooks/normalised_replay.py")}>notebooks/normalised_replay.py</a>, printed
              output in <a href={file("results/normalised_output.txt")}>results/normalised_output.txt</a>, numbers in{" "}
              <a href={file("results/normalised_replay.json")}>results/normalised_replay.json</a>.
            </li>
            <li>
              As shared, its loop right after printing "Late Replay Weights" unpacks two-item pairs into three names, which Python rejects.
              The recorded output continues past it, so the code that ran differed there. The file notes this and is otherwise unchanged.
            </li>
          </ul>
          <a className="btn btn-primary" href="#/unnormalised" style={{ alignSelf: "flex-start" }}>
            Back to run A, unnormalised replay
          </a>
        </div>
      </div>
    </Section>
  );
}
