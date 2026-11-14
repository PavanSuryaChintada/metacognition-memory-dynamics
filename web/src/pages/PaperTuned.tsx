import { figures, NOTEBOOKS, paperTuned } from "../data/results";
import { file } from "../data/derived";
import { Card, Section } from "../components/Layout";
import { SurvivorGrid } from "../components/SurvivorGrid";
import { N } from "../components/ui";

const pt = paperTuned.setup;
const best = paperTuned.best_printed;
const at = (g: number, d: number) => paperTuned.grid.find((c) => c.gamma === g && c.delta === d)!;
const paperCell = at(pt.paper.gamma, pt.paper.delta);
// decay rates are evenly spaced from 0.5 to 1.5, so their mean is 1
const meanDecay = 1;
/** Long-run average per memory from the equation for the total (run C, test 1). */
const predicted = (d: number) => (pt.eta * pt.replay_prob) / (meanDecay + d * pt.n) / pt.n;
/** How far the per-memory average moves around that level with random replay timing (one standard deviation). */
const wobble = (d: number) => {
  const r = 1 - pt.h * (meanDecay + d * pt.n);
  return (pt.h * pt.eta * Math.sqrt(pt.replay_prob * (1 - pt.replay_prob))) / Math.sqrt(1 - r * r) / pt.n;
};
const lowDelta = Math.min(...pt.deltas);
const range = (xs: number[]) => [Math.min(...xs), Math.max(...xs)] as const;
const pctAbove = (v: number) => `${Math.round((v / pt.alive_threshold - 1) * 1000) / 10}%`;

/** Run C, test 2: the same run swept over γ and δ. */
export function PaperTunedView() {
  return (
    <>
      <SweepSection num={1} />
      <DeltaSection num={2} />
      <EdgeSection num={3} />
      <GammaSection num={4} />
      <BestSection num={5} />
      <TunedFiguresSection num={6} />
      <TunedFindingsSection num={7} />
      <TunedCodeSection num={8} />
    </>
  );
}

function SweepSection({ num }: { num: number }) {
  return (
    <Section
      id="sweep"
      label="The sweep"
      num={num}
      headline="25 runs: five values of γ, five of δ."
      lead={`The same ${pt.n} images and settings as test 1, run with five values of γ (diffusion) and five of δ (inhibition), the paper's γ = ${pt.paper.gamma} and δ = ${pt.paper.delta} among them. The seed is reset at the start of every run, so replay fires at the same steps in all 25.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label={`Memories above ${pt.alive_threshold} at step ${pt.steps.toLocaleString("en")}`}
            meta="And the highest activation"
            note={`Rows: γ. Columns: δ. The dashed cell is the paper's values; it reproduces test 1 exactly: ${paperCell.alive} alive, ρ ${paperCell.rho}, highest ${paperCell.max.toFixed(6)}.`}
          >
            <SurvivorGrid large />
          </Card>
        </div>
      </div>
    </Section>
  );
}

function DeltaSection({ num }: { num: number }) {
  return (
    <Section
      id="delta"
      label="Inhibition"
      num={num}
      headline="δ sets the level."
      lead="Test 1's equation for the total predicts the average activation per memory from δ alone: ηp / (mean α + δN) / N. The sweep's numbers follow it, whatever γ is."
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Predicted against measured"
            meta={`Step ${pt.steps.toLocaleString("en")}`}
            note="The highest memory sits a little above the average, because memories with low decay rates end above the rest."
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th className="num">
                      <N>δ</N>
                    </th>
                    <th className="num">Predicted average</th>
                    <th className="num">
                      Highest, across the five <N>γ</N>
                    </th>
                    <th className="num">
                      Above {pt.alive_threshold}, across the five <N>γ</N>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {pt.deltas.map((d) => {
                    const cells = pt.gammas.map((g) => at(g, d));
                    const mx = range(cells.map((c) => c.max));
                    const al = range(cells.map((c) => c.alive));
                    return (
                      <tr key={d}>
                        <td className="num">{d}</td>
                        <td className={`num big${d === lowDelta ? " ember-text" : ""}`}>{predicted(d).toFixed(6)}</td>
                        <td className="num">
                          {mx[0].toFixed(6)} to {mx[1].toFixed(6)}
                        </td>
                        <td className="num">{al[0] === al[1] ? al[0] : `${al[0]} to ${al[1]}`}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
          <p className="muted" data-reveal>
            δ = {lowDelta} is {Math.round(pt.paper.delta / lowDelta)} times below the paper's {pt.paper.delta}, and it is the only value in the
            sweep whose predicted average reaches the {pt.alive_threshold} line.
          </p>
        </div>
      </div>
    </Section>
  );
}

function EdgeSection({ num }: { num: number }) {
  return (
    <Section
      id="edge"
      label="On the line"
      num={num}
      headline={`At δ = ${lowDelta} the average sits exactly on the line.`}
      lead={`${(pt.eta * pt.replay_prob).toFixed(2)} / (1 + ${lowDelta} × ${pt.n}) / ${pt.n} = ${predicted(lowDelta).toFixed(4)}. So for every memory to count as alive, all ${pt.n} have to stay above a line their average sits on.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Mean of the 200" meta={`γ = ${best.gamma}`}>
              <div className="stat-value">{best.mean_alive.toFixed(6)}</div>
              <div className="stat-sub">{pctAbove(best.mean_alive)} above the line at step {pt.steps.toLocaleString("en")}.</div>
            </Card>
            <Card label="Lowest" meta={`γ = ${best.gamma}`} delay={1}>
              <div className="stat-value ember">{best.min_alive.toFixed(6)}</div>
              <div className="stat-sub">{pctAbove(best.min_alive)} above the line.</div>
            </Card>
            <Card label="Highest" meta={`γ = ${best.gamma}`} delay={2}>
              <div className="stat-value">{best.max.toFixed(6)}</div>
              <div className="stat-sub">{pctAbove(best.max)} above the line.</div>
            </Card>
          </div>
          <ul className="points" data-reveal>
            <li>
              <b>Replay fires at random, so the total rises and falls around its average.</b> Worked out from the equation for the total,
              at δ = {lowDelta} the average per memory moves by about {wobble(lowDelta).toFixed(5)} (one standard deviation) around{" "}
              {predicted(lowDelta).toFixed(4)}. At a given step it is about as likely to be below the line as above it.
            </li>
            <li>
              <b>{best.alive} of {pt.n} is this run's count at step {pt.steps.toLocaleString("en")}.</b> At that step the average happened to
              be {pctAbove(best.mean_alive)} above the line. A few replay events either way could have moved every memory under it.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function GammaSection({ num }: { num: number }) {
  const rows = pt.gammas.map((g) => ({ g, ...at(g, lowDelta) }));
  const fewest = rows.reduce((a, b) => (b.alive < a.alive ? b : a));
  const paperRow = rows.find((r) => r.g === pt.paper.gamma)!;
  return (
    <Section
      id="gamma"
      label="Diffusion"
      num={num}
      headline="Less diffusion loses memories."
      lead={`At δ = ${lowDelta} the total is nearly the same whatever γ is; γ decides how far apart the ${pt.n} memories sit. Diffusion pulls them together. With less of it the slow-decaying memories climb and the fast ones drop below the line.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label={
              <>
                <N>δ</N> = {lowDelta}, by <N>γ</N>
              </>
            }
            meta={`Step ${pt.steps.toLocaleString("en")}`}
            note={`The notebook picks γ = ${best.gamma} as best because its rule is ${paperTuned.selection_rule}: γ = ${best.gamma} and γ = ${pt.paper.gamma} both keep ${best.alive}, and ρ is ${best.rho} against ${paperRow.rho}.`}
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th className="num">
                      <N>γ</N>
                    </th>
                    <th className="num">Above {pt.alive_threshold}</th>
                    <th className="num">Highest</th>
                    <th className="num">
                      Spearman <N>ρ</N>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.g}>
                      <td className="num">
                        {r.g}
                        {r.g === pt.paper.gamma && <span className="tag note">paper</span>}
                      </td>
                      <td className="num big">
                        {r.alive} of {pt.n}
                      </td>
                      <td className="num">{r.max.toFixed(6)}</td>
                      <td className="num">{r.rho.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <p className="muted" data-reveal>
            So the paper's γ = {pt.paper.gamma} doesn't need lowering: with δ = {lowDelta} it keeps all {pt.n}. Lowering γ to {fewest.g} keeps{" "}
            {fewest.alive}.
          </p>
        </div>
      </div>
    </Section>
  );
}

function BestSection({ num }: { num: number }) {
  return (
    <Section
      id="best"
      label="Decay order"
      num={num}
      headline={`Ordered by decay rate, from ${best.max.toFixed(6)} down to ${best.min_alive.toFixed(6)}.`}
      lead={`The notebook's chosen run, γ = ${best.gamma} and δ = ${best.delta}. Spearman ρ between decay rate and final activation is ${best.rho}.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="grid-2">
            <Card label="Top ten" meta="As printed">
              <div className="scroll">
                <table>
                  <thead>
                    <tr>
                      <th className="num">Rank</th>
                      <th>Class</th>
                      <th className="num">
                        <N>α</N>
                      </th>
                      <th className="num">Activation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paperTuned.top10_best.map((r) => (
                      <tr key={r.rank}>
                        <td className="num">{r.rank}</td>
                        <td>{r.label}</td>
                        <td className="num">{r.decay.toFixed(4)}</td>
                        <td className="num">{r.activation.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <Card
              label="Above the line by decay rate"
              meta="As printed"
              delay={1}
              note="The last bin prints 39 of 39 because the highest rate, 1.5, sits on its upper edge and the bins leave it out."
            >
              <table>
                <thead>
                  <tr>
                    <th>Decay rate</th>
                    <th className="num">Above {pt.alive_threshold}</th>
                  </tr>
                </thead>
                <tbody>
                  {paperTuned.survival_by_decay_bin_best.map((b) => (
                    <tr key={b.from}>
                      <td>
                        {b.from.toFixed(1)} to {b.to.toFixed(1)}
                      </td>
                      <td className="num">
                        {b.survived} of {b.of}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        </div>
      </div>
    </Section>
  );
}

function TunedFiguresSection({ num }: { num: number }) {
  return (
    <Section id="figure" label="Figures" num={num} headline="The sweep, and the paper's values against the tuned ones.">
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Figure 1 · The sweep"
            meta="Colab output"
            note="The notebook's own figure. Left: memories above 0.001 for each γ and δ, the paper's cell dashed. Middle: Spearman ρ for each. Right: the chosen run's activations; all 200 drop from 1 and settle just above 0 by about step 500."
          >
            <div className="plate">
              <img
                src={figures.tunedSweep}
                alt="Three panels. Left heatmap: 0 survivors everywhere except the δ = 0.001 column, which reads 200, 200, 181, 162 and 143 from γ = 0.2 down to 0.005. Middle heatmap: Spearman ρ between 0.98 and 0.99 in every cell. Right: activation curves for the chosen run falling from 1 to just above 0 by step 500."
              />
            </div>
          </Card>
          <Card
            label="Figure 2 · Paper's values against tuned"
            meta="Colab output"
            note={`The notebook's own figure. Left: γ = ${pt.paper.gamma}, δ = ${pt.paper.delta}; everything is below the line. Right: γ = ${best.gamma}, δ = ${best.delta}; every memory ends between ${best.min_alive} and ${best.max}, just above the dashed 0.001 line, falling with decay rate.`}
          >
            <div className="plate">
              <img
                src={figures.tunedCompare}
                alt="Four panels. Top: activation curves for the paper's values, collapsing at once, and for the tuned values, falling over about 500 steps. Bottom: final activation against decay rate; paper's values give a flat line near 0.00008, tuned values give a falling band from about 0.00118 to 0.00102, just above the 0.001 line."
              />
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function TunedFindingsSection({ num }: { num: number }) {
  return (
    <Section id="findings" label="Findings" num={num} headline="What it shows, and what it doesn't.">
      <div className="grid">
        <div className="indent two-col">
          <div data-reveal>
            <div className="label">Shows</div>
            <ul className="points">
              <li>
                δ decides whether memories can stay above {pt.alive_threshold}. Only δ = {lowDelta}, {Math.round(pt.paper.delta / lowDelta)} times
                below the paper's, does, and the level follows the equation for the total.
              </li>
              <li>
                With δ = {lowDelta}, γ = {pt.paper.gamma} and γ = {best.gamma} keep all {pt.n} at step {pt.steps.toLocaleString("en")}; lower γ keeps
                fewer.
              </li>
              <li>Every run orders the memories by decay rate, ρ between 0.98 and 0.99.</li>
            </ul>
          </div>
          <div data-reveal>
            <div className="label">Doesn't show</div>
            <ul className="points">
              <li>That {best.alive} of {pt.n} holds over time: their average sits on the line, and this is one step of one run.</li>
              <li>That the embeddings' similarity is the reason: the total doesn't depend on the embeddings, and inhibition grows with N.</li>
              <li>
                That γ needs lowering. The notebook's closing text says γ and δ both need cutting about 10 to 20 times; the sweep shows δ
                needed 100 times and γ none.
              </li>
              <li>δ between {lowDelta} and 0.005, or below {lowDelta}; other seeds or images.</li>
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
}

function TunedCodeSection({ num }: { num: number }) {
  return (
    <Section id="run-code" label="The code" num={num} headline="Where these numbers come from.">
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>The executed notebook.</b> <a href={NOTEBOOKS.paperTuned}>notebooks/colab/paper/paper_scale_tuned.ipynb</a>, with its code,
              printed output and figures as they ran in Colab.
            </li>
            <li>
              <b>Recorded copies.</b> Printed output in <a href={file("results/paper_tuned_output.txt")}>results/paper_tuned_output.txt</a>,
              numbers in <a href={file("results/paper_tuned.json")}>results/paper_tuned.json</a>.
            </li>
          </ul>
          <div className="btn-row">
            <a className="btn btn-primary" href={NOTEBOOKS.paperTuned}>
              Open the Colab notebook on GitHub
            </a>
            <a className="btn btn-text" href="#/paper">
              <span aria-hidden="true">←</span> Test 1
            </a>
          </div>
        </div>
      </div>
    </Section>
  );
}
