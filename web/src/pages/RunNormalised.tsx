import {
  figures,
  memories,
  nameOf,
  normalised,
  normalisedStartShares,
  NOTEBOOKS,
  recorded,
  replay,
  ten,
  tenMemories,
} from "../data/results";
import { file, lateShareRange, normalisedFinals } from "../data/derived";
import { Card, Section } from "../components/Layout";
import { RunIntro } from "../components/RunIntro";
import { MiniBars, Outcome } from "../components/Memories";
import { N } from "../components/ui";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
/** Lower-case a memory name mid-sentence, keeping "French" capitalised. */
const lc = (name: string) => (name.startsWith("French") ? name : name.toLowerCase());
const CARD_SCALE = 0.05; // the 4- and 10-memory cards share this bar scale

type View = 4 | 10;

/** Run B: replay scores from embeddings scaled to length 1, run with 4 and with 10 memories. */
export function RunNormalised({ view }: { view: View }) {
  return (
    <>
      <RunIntro
        run="Run B"
        name="Normalised replay"
        headline="Replay spreads, and every memory stays."
        explanation={
          <>
            <p>
              The only change is inside the replay operator: each embedding is scaled to length 1 before the replay scores are computed,
              so a memory's score is the memories' activations weighted by cosine similarity. Decay, diffusion, inhibition, the
              embeddings, the decay rates and the replay coin flips are the same as in the raw arm of the same script.
            </p>
            <p>
              It was run twice: with the four photos used everywhere else on this site, and with ten photos. Each run also repeats the
              raw version side by side, so every comparison below is within one script. Pick one.
            </p>
          </>
        }
        formula={String.raw`F(a) = \mathrm{softmax}\big(\beta\, \hat X^\top \hat X a\big)`}
        formulaNote="X̂ is X with every embedding scaled to length 1, so X̂ᵀX̂ holds cosine similarities. Only F(a) uses it; diffusion already used cosine similarity in both runs."
      />
      <ExperimentCards view={view} />
      {view === 4 ? <FourMemories /> : <TenMemoriesView />}
    </>
  );
}

/* ------------------------------------------------------------------ cards */

function ExperimentCards({ view }: { view: View }) {
  const tenLate = ten.final_activations.normalised;
  return (
    <section className="section" id="experiments">
      <div className="wrap">
        <div className="run-cards experiment-cards">
          <a className={`run-card experiment-card${view === 4 ? " on" : ""}`} href="#/normalised" aria-current={view === 4 ? "page" : undefined}>
            <div className="run-card-head">
              <span className="label">Experiment 1</span>
              {view === 4 && <span className="tag new">Showing</span>}
            </div>
            <h2 className="run-title">4 memories</h2>
            <p className="run-desc">The four photos from run A: beagle, cat, person, French bulldog.</p>
            <MiniBars finals={normalisedFinals} scale={CARD_SCALE} />
            <dl className="run-stats">
              <div>
                <dt>Memories left</dt>
                <dd>
                  {normalised.survivors_above_0_001.normalised} of 4 <span className="dd-sub">raw: {normalised.survivors_above_0_001.unnormalised} of 4</span>
                </dd>
              </div>
              <div>
                <dt>Replay share each</dt>
                <dd>
                  {pct(lateShareRange[0])} to {pct(lateShareRange[1])}
                </dd>
              </div>
            </dl>
          </a>
          <a className={`run-card experiment-card${view === 10 ? " on" : ""}`} href="#/normalised/10" aria-current={view === 10 ? "page" : undefined}>
            <div className="run-card-head">
              <span className="label">Experiment 2</span>
              {view === 10 && <span className="tag new">Showing</span>}
            </div>
            <h2 className="run-title">10 memories</h2>
            <p className="run-desc">The same four plus a horse, a kitten, a landscape, an armchair, a car and a puppy.</p>
            <MiniBars finals={tenLate} scale={CARD_SCALE} items={tenMemories} />
            <dl className="run-stats">
              <div>
                <dt>Memories left</dt>
                <dd>
                  {ten.survivors_above_0_001.normalised} of 10 <span className="dd-sub">raw: {ten.survivors_above_0_001.unnormalised} of 10</span>
                </dd>
              </div>
              <div>
                <dt>Replay share each</dt>
                <dd>
                  near 10% <span className="dd-sub">read off the figure</span>
                </dd>
              </div>
            </dl>
          </a>
        </div>
        <p className="run-cards-note muted small">
          Bars show the normalised arm's activation at step 500 on the same scale, 0 to {CARD_SCALE}, in both cards. Same η, β, p and
          seed in both; the first four photos and their decay rates are the same.
        </p>
      </div>
    </section>
  );
}

/* --------------------------------------------------------------- 4 memories */

const raw4 = normalised.final_activations.unnormalised;
const norm4 = normalised.final_activations.normalised;
const late4 = normalised.replay_weight_mean_last_10pct;
const beta = normalised.setup.beta;
const startShares = normalisedStartShares(beta);
const finalsSum4 = Object.values(norm4).reduce((a, b) => a + b, 0);
const byDecay4 = [...memories].sort((a, b) => a.decay_rate - b.decay_rate);
const top4 = byDecay4[0];
const sim = replay.cosine_similarity.flatMap((row, i) => row.filter((_, j) => j !== i));

function FourMemories() {
  return (
    <>
      <div className="wrap">
        <Outcome
          finals={normalisedFinals}
          label={`Figure 1 · Activation after ${normalised.setup.steps} steps, 4 memories`}
          meta={`Normalised arm · seed ${normalised.setup.seed} · η = ${normalised.setup.eta}, p = ${normalised.setup.replay_prob}`}
        />
      </div>
      <ChangeSection num={1} />
      <Result4Section num={2} />
      <SpreadSection num={3} />
      <Figure4Section num={4} />
      <Findings4Section num={5} />
      <Code4Section num={6} />
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
      lead="The comparison notebook runs both versions back to back on the same encoded photos. This is everything that differs between them."
    >
      <div className="grid">
        <div className="indent stack">
          <div className="grid-2">
            <Card label="Raw arm" meta="As the paper writes F(a)">
              <pre className="code-light">
                <code>{"s = X @ (X.T @ a)\nF = softmax(beta * s)"}</code>
              </pre>
            </Card>
            <Card label="Normalised arm" meta="Each row of X scaled to length 1" delay={1}>
              <pre className="code-light">
                <code>{"X_n = normalize(X, dim=1)\ns = X_n @ (X_n.T @ a)\nF = softmax(beta * s)"}</code>
              </pre>
            </Card>
          </div>
          <Card label="The same in both arms" meta="normalised_metacognition.ipynb">
            <ul className="points compact">
              <li>The four I-JEPA embeddings, encoded once.</li>
              <li>Decay rates {normalised.decay_rates_printed.map((r) => r.toFixed(2)).join(", ")}.</li>
              <li>
                <N>η</N> = {normalised.setup.eta}, <N>β</N> = {beta}, <N>γ</N> = {normalised.setup.gamma}, <N>δ</N> = {normalised.setup.delta},
                h = {normalised.setup.h}, p = {normalised.setup.replay_prob}, {normalised.setup.steps} steps.
              </li>
              <li>
                The replay coin flips: the seed is reset before each arm, so replay fired at the same {normalised.replay_events.normalised} steps in
                both.
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

function Result4Section({ num }: { num: number }) {
  return (
    <Section
      id="result"
      label="The result"
      num={num}
      headline="All four memories stay, in decay order."
      lead="Final activations and replay shares for both arms of the comparison notebook."
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
            <Card label="Highest at the end" meta={nameOf(top4.label)} delay={2}>
              <div className="stat-value">{norm4[top4.label].toFixed(4)}</div>
              <div className="stat-sub">
                The memory with the slowest decay, <N>α</N> = {top4.decay_rate.toFixed(2)}.
              </div>
            </Card>
          </div>
          <Card
            label="Activation at step 500"
            meta="Sorted by decay rate"
            note={`In this notebook the seed is reset after the decay rates are drawn, so the coin flips start four draws earlier than in the original notebook. That is why the raw arm ends at ${raw4.person} here and at ${recorded.single_run.final_activations.person} there.`}
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
                {byDecay4.map((m) => (
                  <tr key={m.label}>
                    <td>{m.name}</td>
                    <td className="num">{m.decay_rate.toFixed(2)}</td>
                    <td className="num big">{raw4[m.label] > 0 ? raw4[m.label].toFixed(6) : "0"}</td>
                    <td className="num big ember-text">{norm4[m.label].toFixed(6)}</td>
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
      lead={`With unit-length embeddings, a replay score is a sum of cosine similarities weighted by activation. Between these four photos the cosine similarities run from ${Math.min(...sim).toFixed(2)} to ${Math.max(...sim).toFixed(2)}, which keeps the scores small, so the softmax never concentrates.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Replay share by memory"
            meta="Normalised arm"
            note={
              <>
                Start: worked out from the measured cosine similarities with every activation at 1, using the notebook's own formula. The
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
                    <td className="num big">{pct(late4.normalised[l])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <p className="muted" data-reveal>
            Why it evens out: each score is at most the sum of all activations. At step {normalised.setup.steps} that sum is {finalsSum4.toFixed(3)}, so{" "}
            <N>β</N> times any score is at most {(beta * finalsSum4).toFixed(2)}, and no memory's share can be more than{" "}
            {Math.exp(beta * finalsSum4).toFixed(1)} times another's. The recorded shares sit between {pct(lateShareRange[0])} and{" "}
            {pct(lateShareRange[1])}.
          </p>
        </div>
      </div>
    </Section>
  );
}

function Figure4Section({ num }: { num: number }) {
  return (
    <Section id="figure" label="Figure" num={num} headline="Both arms, step by step.">
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Figure 2 · Raw and normalised replay, 4 memories"
            meta="Colab output"
            note={
              <>
                The notebook's own figure. Top row: raw arm. Bottom row: normalised arm. Its labels are from that run: <code>labrador</code> is
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

function Findings4Section({ num }: { num: number }) {
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
                {byDecay4.map((m) => `${lc(m.name)} ${norm4[m.label].toFixed(4)}`).join(", ")}.
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

function Code4Section({ num }: { num: number }) {
  return (
    <Section id="run-code" label="The code" num={num} headline="Where these numbers come from.">
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>The executed notebook.</b>{" "}
              <a href={NOTEBOOKS.normalised4}>notebooks/colab/normalised/normalised_metacognition.ipynb</a>, with its code, printed output and
              figure as they ran in Colab.
            </li>
            <li>
              <b>Recorded copies.</b> Printed output in <a href={file("results/normalised_output.txt")}>results/normalised_output.txt</a>, numbers in{" "}
              <a href={file("results/normalised_replay.json")}>results/normalised_replay.json</a>.
            </li>
          </ul>
          <div className="btn-row">
            <a className="btn btn-primary" href={NOTEBOOKS.normalised4}>
              Open the Colab notebook on GitHub
            </a>
            <a className="btn btn-text" href="#/normalised/10">
              Next: 10 memories <span aria-hidden="true">→</span>
            </a>
          </div>
        </div>
      </div>
    </Section>
  );
}

/* -------------------------------------------------------------- 10 memories */

const raw10 = ten.final_activations.unnormalised;
const norm10 = ten.final_activations.normalised;
const tenByDecay = ten.order_by_decay_as_printed.map((l) => tenMemories.find((m) => m.label === l)!);
const tenTop = tenByDecay[0];
const rawWinner10 = Object.keys(raw10)[0];
// Neighbours in the notebook's decay order (np.argsort on the unrounded rates) whose final activations come out the other way round.
const pairs = tenByDecay.slice(0, -1).map((m, i) => [m, tenByDecay[i + 1]] as const);
const swaps = pairs.filter(([a, b]) => norm10[a.label] < norm10[b.label]);
const swapped = new Set(swaps.flatMap(([a, b]) => [a.label, b.label]));
const ties = pairs.filter(([a, b]) => a.decay_rate === b.decay_rate);
const maxGap = Math.max(...swaps.map(([a, b]) => norm10[b.label] - norm10[a.label]));
const rankByFinal = [...tenByDecay].sort((a, b) => norm10[b.label] - norm10[a.label]).map((m) => m.label);

function TenMemoriesView() {
  return (
    <>
      <div className="wrap">
        <Outcome
          finals={norm10}
          items={tenMemories}
          label={`Figure 1 · Activation after ${ten.setup.steps} steps, 10 memories`}
          meta={`Normalised arm · seed ${ten.setup.seed} · η = ${ten.setup.eta}, p = ${ten.setup.replay_prob}`}
        />
      </div>
      <Result10Section num={1} />
      <OrderSection num={2} />
      <Figure10Section num={3} />
      <Findings10Section num={4} />
      <Code10Section num={5} />
    </>
  );
}

function Result10Section({ num }: { num: number }) {
  return (
    <Section
      id="result"
      label="The result"
      num={num}
      headline="All ten memories stay with normalised replay."
      lead="Ten photos, the same parameters and seed, raw and normalised replay side by side. In the raw arm one memory is left; in the normalised arm all ten are."
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Memories left" meta="Above 0.001 at step 500">
              <div className="stat-value ember">
                {ten.survivors_above_0_001.normalised}
                <span className="unit">of 10</span>
              </div>
              <div className="stat-sub">The raw arm keeps {ten.survivors_above_0_001.unnormalised} of 10.</div>
            </Card>
            <Card label="Raw arm's only memory" meta={nameOf(rawWinner10)} delay={1}>
              <div className="stat-value">{raw10[rawWinner10].toFixed(4)}</div>
              <div className="stat-sub">The {nameOf(rawWinner10).toLowerCase()}, which also has the slowest decay here.</div>
            </Card>
            <Card label="Highest at the end" meta={nameOf(tenTop.label)} delay={2}>
              <div className="stat-value">{norm10[tenTop.label].toFixed(4)}</div>
              <div className="stat-sub">
                Normalised arm. Slowest decay, <N>α</N> = {tenTop.decay_rate.toFixed(2)}.
              </div>
            </Card>
          </div>
          <Card
            label="Activation at step 500"
            meta="In the notebook's decay order"
            note="The notebook prints DEAD for any memory at or below 0.001, so the raw arm's other nine values are known only to be at most 0.001."
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Memory</th>
                    <th>Label in the run</th>
                    <th className="num">
                      Decay rate <N>α</N>
                    </th>
                    <th className="num">Raw arm</th>
                    <th className="num">Normalised arm</th>
                  </tr>
                </thead>
                <tbody>
                  {tenByDecay.map((m) => (
                    <tr key={m.label}>
                      <td>{m.name}</td>
                      <td className="muted">
                        <code>{m.label_in_run}</code>
                      </td>
                      <td className="num">{m.decay_rate.toFixed(2)}</td>
                      <td className="num">{raw10[m.label] !== undefined ? raw10[m.label].toFixed(4) : "≤ 0.001"}</td>
                      <td className="num big ember-text">{norm10[m.label].toFixed(4)}</td>
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

const COUNT = ["no", "one", "two", "three", "four", "five"];

function OrderSection({ num }: { num: number }) {
  return (
    <Section
      id="order"
      label="Decay order"
      num={num}
      headline={`Close to decay order, with ${COUNT[swaps.length]} swaps.`}
      lead={`Listed the way the notebook lists them, by decay rate, the normalised arm's final activations fall nearly in step. ${swaps.length} neighbouring pairs come out the other way round, the widest by ${maxGap.toFixed(4)}.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Rank by decay rate against rank by final activation"
            meta="Normalised arm"
            note={
              <>
                {ties.length} pairs print the same decay rate:{" "}
                {ties.map(([a, b]) => `${lc(a.name)} and ${lc(b.name)} (${a.decay_rate.toFixed(2)})`).join(", ")}. The notebook orders its
                table by the unrounded rates, which it doesn't print; this table keeps that order. Its own check prints "Decay rate predicts
                activation ranking: YES" because it allows a slack of 0.001 between neighbours, and every gap here is smaller than that.
              </>
            }
          >
            <div className="scroll">
              <table className="rank-table">
                <thead>
                  <tr>
                    <th className="num">By decay</th>
                    <th>Memory</th>
                    <th className="num">
                      <N>α</N>
                    </th>
                    <th className="num">Final</th>
                    <th className="num">By final</th>
                  </tr>
                </thead>
                <tbody>
                  {tenByDecay.map((m, i) => {
                    const rf = rankByFinal.indexOf(m.label) + 1;
                    return (
                      <tr key={m.label} className={swapped.has(m.label) ? "swap" : undefined}>
                        <td className="num">{i + 1}</td>
                        <td>
                          {m.name}
                          {swapped.has(m.label) && <span className="tag ember">Swapped</span>}
                        </td>
                        <td className="num">{m.decay_rate.toFixed(2)}</td>
                        <td className="num big">{norm10[m.label].toFixed(4)}</td>
                        <td className="num">{rf}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
          <ul className="points" data-reveal>
            {swaps.map(([a, b]) => (
              <li key={a.label}>
                <b>
                  {a.name} and {lc(b.name)}.
                </b>{" "}
                {a.decay_rate === b.decay_rate ? (
                  <>
                    Both print <N>α</N> = {a.decay_rate.toFixed(2)}. The notebook lists {lc(a.name)} first, so its unrounded rate is the lower
                    one, but it ends lower, {norm10[a.label].toFixed(4)} against {norm10[b.label].toFixed(4)}.
                  </>
                ) : (
                  <>
                    {a.name} decays slower (<N>α</N> = {a.decay_rate.toFixed(2)} against {b.decay_rate.toFixed(2)}) but ends lower,{" "}
                    {norm10[a.label].toFixed(4)} against {norm10[b.label].toFixed(4)}.
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Section>
  );
}

function Figure10Section({ num }: { num: number }) {
  return (
    <Section id="figure" label="Figure" num={num} headline="Both arms, step by step.">
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Figure 2 · Raw and normalised replay, 10 memories"
            meta="Colab output"
            note={
              <>
                The notebook's own figure. Top row: raw arm, where one replay-weight line sits at 1 for the whole run; the legend colours
                it as the landscape. Bottom row: normalised arm, where the replay weights start uneven and settle near 0.10 each, read off
                the panel. Its labels are from that run: <code>labrador</code> is the beagle, <code>bulldog</code> the French bulldog and{" "}
                <code>coffee</code> the armchair photo.
              </>
            }
          >
            <div className="plate">
              <img
                src={figures.ten}
                alt="Four panels. Top: raw replay, only the landscape stays near 0.2 and one memory takes all replay weight. Bottom: normalised replay, all ten activations settle near 0.01 to 0.02 and the replay weights converge near 0.10."
              />
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function Findings10Section({ num }: { num: number }) {
  return (
    <Section id="findings" label="Findings" num={num} headline="What it shows, and what it doesn't.">
      <div className="grid">
        <div className="indent two-col">
          <div data-reveal>
            <div className="label">Shows</div>
            <ul className="points">
              <li>
                With ten memories, raw replay again leaves one memory, the {nameOf(rawWinner10).toLowerCase()}. Here that memory also has the
                slowest decay; in the 4-memory run it did not.
              </li>
              <li>Normalised replay keeps all ten.</li>
              <li>
                The normalised arm's final activations follow decay rate closely, with {swaps.length} neighbouring pairs reversed by less than
                0.001.
              </li>
            </ul>
          </div>
          <div data-reveal>
            <div className="label">Doesn't show</div>
            <ul className="points">
              <li>Replay shares as numbers: this notebook doesn't print them, so they are read off its figure.</li>
              <li>Why the raw arm picked the landscape. The run didn't print the replay scores or the embedding lengths.</li>
              <li>Anything statistical: one seed, and one set of ten photos.</li>
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
}

function Code10Section({ num }: { num: number }) {
  return (
    <Section id="run-code" label="The code" num={num} headline="Where these numbers come from.">
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>The executed notebook.</b> <a href={NOTEBOOKS.normalised10}>notebooks/colab/normalised/metacognition_10_mem.ipynb</a>, with
              its code, printed output and figure as they ran in Colab.
            </li>
            <li>
              <b>Recorded copies.</b> Printed output in{" "}
              <a href={file("results/normalised_10_memories_output.txt")}>results/normalised_10_memories_output.txt</a>, numbers in{" "}
              <a href={file("results/normalised_10_memories.json")}>results/normalised_10_memories.json</a>.
            </li>
          </ul>
          <div className="btn-row">
            <a className="btn btn-primary" href={NOTEBOOKS.normalised10}>
              Open the Colab notebook on GitHub
            </a>
            <a className="btn btn-text" href="#/normalised">
              <span aria-hidden="true">←</span> Back to 4 memories
            </a>
          </div>
        </div>
      </div>
    </Section>
  );
}

