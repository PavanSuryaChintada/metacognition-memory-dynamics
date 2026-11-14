import {
  figures,
  forgetting,
  memories,
  nameOf,
  normalised,
  normalisedStartShares,
  NOTEBOOKS,
  recorded,
  replay,
  ten,
  tenMemories,
  forgettingMemories,
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

type View = 4 | 10 | "beyond";

/** Run B: replay scores from embeddings scaled to length 1, run with 4 and with 10 memories, plus tests beyond the paper. */
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
              raw version side by side, so every comparison below is within one script. A third box holds tests beyond the paper,
              starting with new memories added partway through a run. Pick one.
            </p>
          </>
        }
        formula={String.raw`F(a) = \mathrm{softmax}\big(\beta\, \hat X^\top \hat X a\big)`}
        formulaNote="X̂ is X with every embedding scaled to length 1, so X̂ᵀX̂ holds cosine similarities. Only F(a) uses it; diffusion already used cosine similarity in both runs."
      />
      <ExperimentCards view={view} />
      {view === 4 ? <FourMemories /> : view === 10 ? <TenMemoriesView /> : <BeyondView />}
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
          <a
            className={`run-card experiment-card${view === "beyond" ? " on" : ""}`}
            href="#/normalised/beyond"
            aria-current={view === "beyond" ? "page" : undefined}
          >
            <div className="run-card-head">
              <span className="label">Experiment 3</span>
              {view === "beyond" && <span className="tag new">Showing</span>}
            </div>
            <h2 className="run-title">Tests beyond the paper</h2>
            <p className="run-desc">
              New memories partway through a run: five stored at step 0, five more added at step {forgetting.setup.inject_at}.
            </p>
            <Timeline />
            <dl className="run-stats">
              <div>
                <dt>Old memories left</dt>
                <dd>
                  {forgetting.survivors.old} of {forgetting.survivors.old_total}{" "}
                  <span className="dd-sub">the one lost is a random vector, not a photo</span>
                </dd>
              </div>
              <div>
                <dt>New memories left</dt>
                <dd>
                  {forgetting.survivors.new} of {forgetting.survivors.new_total}
                </dd>
              </div>
            </dl>
          </a>
        </div>
        <p className="run-cards-note muted small">
          In the first two cards, bars show the normalised arm's activation at step 500 on the same scale, 0 to {CARD_SCALE}. Same η, β,
          p and seed in both; the first four photos and their decay rates are the same.
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

/* ------------------------------------------------------- tests beyond the paper */

const fg = forgetting.setup;
const oldMems = forgettingMemories.filter((m) => m.group === "old");
const newMems = forgettingMemories.filter((m) => m.group === "new");
const randomVec = forgettingMemories.find((m) => m.no_photo)!;
const oldPhotos = oldMems.filter((m) => !m.no_photo);
const sumBefore = oldMems.reduce((s, m) => s + m.at_250, 0);
const sumAfter = sumBefore + newMems.length * fg.new_memory_start_activation;
const range = (xs: number[]) => [Math.min(...xs), Math.max(...xs)] as const;
const oldPhotoFinal = range(oldPhotos.map((m) => m.final));
const newFinal = range(newMems.map((m) => m.final));
const pctChange = forgetting.old_change_250_to_500_pct_printed;
const photoDrops = range(oldPhotos.map((m) => -pctChange[m.label]));
const topEnd = [...forgettingMemories].sort((a, b) => b.final - a.final)[0];
const minDecayAll = Math.min(...forgettingMemories.map((m) => m.decay_rate_printed));
// an old photo and a new one that print the same decay rate: same α, different age
const pairOld = oldPhotos.find((o) => newMems.some((n) => n.decay_rate_printed === o.decay_rate_printed))!;
const pairNew = newMems.find((n) => n.decay_rate_printed === pairOld.decay_rate_printed)!;
const f4 = (v: number) => v.toFixed(4);

/** Which memories exist when: five from step 0, five more from the injection step. */
function Timeline({ large = false }: { large?: boolean }) {
  const at = (fg.inject_at / fg.steps) * 100;
  return (
    <div
      className={`timeline${large ? " timeline-large" : ""}`}
      role="img"
      aria-label={`Five old memories from step 0 to ${fg.steps}; five new memories from step ${fg.inject_at} to ${fg.steps}.`}
    >
      <div className="tl-row">
        <span className="tl-name">5 old</span>
        <span className="tl-track">
          <span className="tl-bar" style={{ left: 0, right: 0 }} />
        </span>
      </div>
      <div className="tl-row">
        <span className="tl-name">5 new</span>
        <span className="tl-track">
          <span className="tl-bar tl-new" style={{ left: `${at}%`, right: 0 }} />
        </span>
      </div>
      <div className="tl-row">
        <span className="tl-name">step</span>
        <span className="tl-ticks">
          <span>0</span>
          <span>{fg.inject_at}</span>
          <span>{fg.steps}</span>
        </span>
      </div>
    </div>
  );
}

function Thumb({ m }: { m: (typeof forgettingMemories)[number] }) {
  return m.thumb ? (
    <img className="thumb" src={m.thumb} alt="" width={48} height={36} />
  ) : (
    <span className="thumb thumb-none" aria-hidden="true">
      no photo
    </span>
  );
}

function BeyondView() {
  return (
    <>
      <TestSection num={1} />
      <BeyondResultSection num={2} />
      <OldMemoriesSection num={3} />
      <NewVsOldSection num={4} />
      <VerdictLineSection num={5} />
      <BeyondFigureSection num={6} />
      <BeyondFindingsSection num={7} />
      <BeyondCodeSection num={8} />
    </>
  );
}

function TestSection({ num }: { num: number }) {
  return (
    <Section
      id="test"
      label="The test"
      num={num}
      headline="Five memories, then five more halfway through."
      lead={`Every other run on this site stores all of its memories at step 0. Here five are stored at step 0 and run for ${fg.inject_at} steps. Then five new ones are added at activation ${fg.new_memory_start_activation}, and all ten run on to step ${fg.steps}.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Timeline"
            meta="5_5_forgetting_test.ipynb"
            note={
              <>
                The horse photo's link returned 404 three times, so the notebook put a random vector (<code>torch.randn(1280)</code>) in its
                place. It runs as a memory labelled <code>horse</code>, but it is not a photo, and this page calls it the random vector.
              </>
            }
          >
            <Timeline large />
            <ul className="points compact" style={{ marginTop: 18 }}>
              <li>
                <b>Old, from step 0:</b> {oldMems.map((m) => lc(m.name)).join(", ")}.
              </li>
              <li>
                <b>New, from step {fg.inject_at}:</b> {newMems.map((m) => lc(m.name)).join(", ")}, each starting at{" "}
                {fg.new_memory_start_activation}.
              </li>
              <li>
                Normalised replay, as in the rest of run B. <N>η</N> = {fg.eta}, <N>β</N> = {fg.beta}, <N>γ</N> = {fg.gamma},{" "}
                <N>δ</N> = {fg.delta}, h = {fg.h}, p = {fg.replay_prob}, seed {fg.seed}.
              </li>
            </ul>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function BeyondResultSection({ num }: { num: number }) {
  return (
    <Section
      id="result"
      label="The result"
      num={num}
      headline={`${forgetting.survivors.old} of ${forgetting.survivors.old_total} old memories and ${forgetting.survivors.new} of ${forgetting.survivors.new_total} new ones are left.`}
      lead={`Status as the notebook printed it at step ${fg.steps}: above ${fg.alive_threshold} counts as left. Each memory's activation just before the new ones arrived, and at the end.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Old memories left" meta={`Above ${fg.alive_threshold}`}>
              <div className="stat-value ember">
                {forgetting.survivors.old}
                <span className="unit">of {forgetting.survivors.old_total}</span>
              </div>
              <div className="stat-sub">
                The one below is the random vector, at {randomVec.final.toFixed(6)}. All four photos stay.
              </div>
            </Card>
            <Card label="New memories left" meta={`Added at step ${fg.inject_at}`} delay={1}>
              <div className="stat-value">
                {forgetting.survivors.new}
                <span className="unit">of {forgetting.survivors.new_total}</span>
              </div>
              <div className="stat-sub">
                They end between {f4(newFinal[0])} and {f4(newFinal[1])}.
              </div>
            </Card>
            <Card label="Highest at the end" meta={topEnd.name} delay={2}>
              <div className="stat-value">{f4(topEnd.final)}</div>
              <div className="stat-sub">
                New, {topEnd.decay_rate_printed === minDecayAll ? "with the slowest decay in the run" : "decay rate"}, <N>α</N> ={" "}
                {topEnd.decay_rate_printed.toFixed(2)}.
              </div>
            </Card>
          </div>
          <Card
            label={`Activation at step ${fg.inject_at} and step ${fg.steps}`}
            meta="In the notebook's order"
            note={`Step ${fg.inject_at} is the activation printed just before the new memories were added; new memories start at ${fg.new_memory_start_activation}. Change is the notebook's own column, from step ${fg.inject_at} to step ${fg.steps}.`}
          >
            <div className="scroll">
              <table className="thumb-table">
                <thead>
                  <tr>
                    <th>Memory</th>
                    <th>Group</th>
                    <th className="num">
                      <N>α</N>
                    </th>
                    <th className="num">Step {fg.inject_at}</th>
                    <th className="num">Step {fg.steps}</th>
                    <th className="num">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {forgettingMemories.map((m) => (
                    <tr key={m.label} className={m.status === "DEAD" ? "gone" : undefined}>
                      <td>
                        <span className="thumb-name">
                          <Thumb m={m} />
                          <span>
                            {m.name}
                            {m.label_in_run !== m.label && (
                              <span className="muted small">
                                {" "}
                                <code>{m.label_in_run}</code>
                              </span>
                            )}
                          </span>
                        </span>
                      </td>
                      <td>{m.group === "old" ? "Old" : "New"}</td>
                      <td className="num">{m.decay_rate_printed.toFixed(2)}</td>
                      <td className="num">{m.group === "old" ? m.at_250.toFixed(6) : `${m.at_250} (added)`}</td>
                      <td className={`num big${m.group === "new" ? " ember-text" : ""}`}>{m.final.toFixed(6)}</td>
                      <td className="num">
                        {m.group === "old" ? `${forgetting.old_change_250_to_500_pct_printed[m.label].toFixed(1)}%` : "—"}
                      </td>
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

function OldMemoriesSection({ num }: { num: number }) {
  return (
    <Section
      id="old"
      label="Old memories"
      num={num}
      headline={`The old photos fall ${Math.round(photoDrops[0])}% to ${Math.round(photoDrops[1])}% after step ${fg.inject_at}.`}
      lead={`That is the printed change from just before the new memories arrived to step ${fg.steps}; the random vector falls ${-pctChange[randomVec.label]}%. The run has no copy without the new memories, so the change also holds whatever decay would have happened anyway. The figure shows when it happens.`}
    >
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>Activations, figure top left.</b> Before step {fg.inject_at} the old memories fall slowly. After it they bend down
              sharply: the random vector drops to 0 almost at once, and within about 100 steps the photos are close to 0 on the panel's
              scale. The four photos end between {f4(oldPhotoFinal[0])} and {f4(oldPhotoFinal[1])}.
            </li>
            <li>
              <b>Replay weights, top right.</b> At step {fg.inject_at} the old memories' weights drop from between 0.15 and 0.24 to between 0
              and 0.11, then climb back to between 0.08 and 0.11 by step {fg.steps}. Read off the panel; the notebook doesn't print them.
            </li>
            <li>
              <b>Inhibition.</b> Summed activation goes from {sumBefore.toFixed(3)} just before step {fg.inject_at} to {sumAfter.toFixed(3)}{" "}
              just after, about {Math.round(sumAfter / sumBefore)} times. The inhibition term <N>δ</N>(1ᵀa) is proportional to that sum and
              takes the same amount off every memory, so it weighs most on the ones that are already small. The run didn't log the forces,
              so how much each term contributed isn't recorded.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function NewVsOldSection({ num }: { num: number }) {
  return (
    <Section
      id="new-old"
      label="New against old"
      num={num}
      headline={`New memories end higher, but they are ${fg.steps - fg.inject_at} steps younger.`}
      lead={`At step ${fg.steps} every new memory is above every old one. The new ones started at ${fg.new_memory_start_activation} and have had ${fg.steps - fg.inject_at} steps to decay; the old ones have had ${fg.steps}.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="grid-2">
            {[pairOld, pairNew].map((m, i) => (
              <Card
                key={m.label}
                label={m.group === "old" ? "Old, from step 0" : `New, from step ${fg.inject_at}`}
                meta={<>Same <N>α</N> = {m.decay_rate_printed.toFixed(2)}</>}
                delay={i}
              >
                <div className="pair-memory">
                  <Thumb m={m} />
                  <span>{m.name}</span>
                </div>
                <div className={`stat-value${m.group === "new" ? " ember" : ""}`}>{f4(m.final)}</div>
                <div className="stat-sub">
                  At step {fg.steps}, after {fg.steps - m.born} steps.
                </div>
              </Card>
            ))}
          </div>
          <p className="muted" data-reveal>
            Same decay rate, and the younger one ends {(pairNew.final / pairOld.final).toFixed(1)} times higher. Without a run that adds
            the same five memories at step 0, being younger and arriving late can't be told apart. In the figure the two slowest new
            memories are still falling at step {fg.steps}.
          </p>
        </div>
      </div>
    </Section>
  );
}

function VerdictLineSection({ num }: { num: number }) {
  return (
    <Section
      id="verdict-line"
      label="The printed verdict"
      num={num}
      headline="The one old memory lost is the random vector."
      lead={
        <>
          The notebook ends with “{forgetting.verdict_printed.split(".")[0]}.” and “Total old-memory activation:{" "}
          {forgetting.old_total_printed.at_250} → {forgetting.old_total_printed.final} ({forgetting.old_total_printed.change_pct.toFixed(1)}
          %)”. Both include the random vector.
        </>
      }
    >
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>The memory that died isn't a photo.</b> It is the random vector the notebook used when the horse link failed. Every old
              memory that is a photo ends above {fg.alive_threshold}.
            </li>
            <li>
              <b>A random vector has almost nothing in common with the photos.</b> In 1,280 dimensions its cosine similarity with any fixed
              vector is close to 0, with a spread of about 1/√1280 ≈ {(1 / Math.sqrt(1280)).toFixed(3)}. Its replay score therefore gets
              almost nothing from the new memories. In the figure its replay weight falls to about 0 at step {fg.inject_at} and stays the
              lowest, and its activation drops to 0 almost at once.
            </li>
            <li>
              <b>The test needs a photo there to say anything about forgetting a horse.</b> With a working link, that slot would hold a
              real I-JEPA embedding; how it would end wasn't run.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function BeyondFigureSection({ num }: { num: number }) {
  return (
    <Section id="figure" label="Figure" num={num} headline="The run, step by step.">
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Figure · Five old memories, five new at step 250"
            meta="Colab output"
            note={
              <>
                The notebook's own figure. Top left: activations; solid lines are old memories, dashed lines new, and the red line is step{" "}
                {fg.inject_at}. Top right: replay weights. Bottom left: each old memory at step {fg.inject_at} and step {fg.steps}. Bottom
                right: every memory at step {fg.steps}. Its labels are from that run: <code>labrador</code> is the beagle,{" "}
                <code>bulldog</code> the French bulldog, <code>coffee</code> the armchair, and <code>horse</code> the random vector.
              </>
            }
          >
            <div className="plate">
              <img
                src={figures.forgetting}
                alt="Four panels. Top left: the five old activations fall slowly, then drop to near zero soon after step 250, when five new memories start at 1 and decay. Top right: old memories' replay weights drop sharply at step 250 and recover to about 0.1; new ones start between 0.12 and 0.22 and settle near 0.1. Bottom left: old memories at step 250 and step 500. Bottom right: all ten at step 500, the five new ones highest."
              />
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function BeyondFindingsSection({ num }: { num: number }) {
  return (
    <Section id="findings" label="Findings" num={num} headline="What it shows, and what it doesn't.">
      <div className="grid">
        <div className="indent two-col">
          <div data-reveal>
            <div className="label">Shows</div>
            <ul className="points">
              <li>
                Adding five memories at activation {fg.new_memory_start_activation} changes the old ones' course at once: their replay
                weights drop and their activations fall to near zero soon after step {fg.inject_at}.
              </li>
              <li>All four old photo memories are still above {fg.alive_threshold} at step {fg.steps}.</li>
              <li>At step {fg.steps} every new memory is above every old one.</li>
            </ul>
          </div>
          <div data-reveal>
            <div className="label">Doesn't show</div>
            <ul className="points">
              <li>How much of the old memories' drop the new memories cause: there is no run without them.</li>
              <li>Whether the new memories end higher because of replay or because they are younger.</li>
              <li>Forgetting of a real horse photo: that slot held a random vector.</li>
              <li>Not tested: new memories starting below 1, or extra replay for old memories after new ones arrive.</li>
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
}

function BeyondCodeSection({ num }: { num: number }) {
  return (
    <Section id="run-code" label="The code" num={num} headline="Where these numbers come from.">
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>The executed notebook.</b>{" "}
              <a href={NOTEBOOKS.forgetting}>notebooks/colab/normalised/beyond_paper/5_5_forgetting_test.ipynb</a>, with its code, printed
              output and figure as they ran in Colab.
            </li>
            <li>
              <b>Recorded copies.</b> Printed output in{" "}
              <a href={file("results/forgetting_5_plus_5_output.txt")}>results/forgetting_5_plus_5_output.txt</a>, numbers in{" "}
              <a href={file("results/forgetting_5_plus_5.json")}>results/forgetting_5_plus_5.json</a>.
            </li>
          </ul>
          <div className="btn-row">
            <a className="btn btn-primary" href={NOTEBOOKS.forgetting}>
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
