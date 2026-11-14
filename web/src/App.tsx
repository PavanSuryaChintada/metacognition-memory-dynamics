import type { CSSProperties } from "react";
import { AUTHOR, AUTHOR_GITHUB, figures, memories, nameOf, recorded, REPO } from "./data/results";
import { Card, Chapter, Header, Mark, Section } from "./components/Layout";
import { HeroArt } from "./components/HeroArt";
import { Outcome } from "./components/Memories";
import { FORCES, ForcesScrolly } from "./components/Forces";
import { MemoryStack } from "./components/MemoryStack";
import { Flow } from "./components/Flow";
import { Pipeline } from "./components/Pipeline";
import { DriveChart } from "./components/DriveChart";
import { Tex } from "./components/Tex";
import { useEnterProgress, useRevealAll } from "./hooks";

const run = recorded.single_run;
const sweepA = recorded.regimes.sweep_a.runs;
const sweepB = recorded.regimes.sweep_b.runs;
const decays = recorded.memories.map((m) => m.decay_rate);
const minDecay = Math.min(...decays);
const slowest = recorded.memories[decays.indexOf(minDecay)];
const survivor = memories.find((m) => m.final > 0)!;
const startEnergy = 0.5 * decays.reduce((s, d) => s + d, 0) + 0.5 * recorded.base_params.delta * decays.length ** 2;
const top = run.retrieval.top[0];
const impliedCosine = top.score / run.final_activations[top.label];
const startNorm = Math.sqrt(recorded.memories.length);
const drive = (r: { eta: number; replay_prob: number }) => +(r.eta * r.replay_prob).toFixed(4);
const bound = (eta: number) => Math.max(startNorm, eta / minDecay);
const file = (path: string) => `${REPO}/blob/main/${path}`;
const N = ({ children }: { children: string }) => <span className="nocase">{children}</span>;
const d = (n: number) => ({ "--d": n }) as CSSProperties;

export function App() {
  useRevealAll();
  return (
    <>
      <Header />
      <main>
        <Hero />
        <ModelChapter />
        <MemoriesSection />
        <HowItRunsSection />
        <Chapter id="results">Results</Chapter>
        <SingleRunSection />
        <RetrievalSection />
        <RegimesSection />
        <TheorySection />
        <PaperSection />
        <FindingsSection />
        <ReproduceSection />
      </main>
      <DarkClose />
    </>
  );
}

function Hero() {
  return (
    <section className="hero" id="top">
      <HeroArt />
      <div className="wrap hero-inner">
        <p className="label rise" style={d(0)}>
          A replication on real embeddings
        </p>
        <h1 className="display rise" style={d(1)}>
          Memory dynamics on real perception.
        </h1>
        <p className="hero-sub rise" style={d(2)}>
          Ghosh, Jaiswal and Gupta model memory as activations that decay, spread across a similarity graph, and get
          reinforced by replay. Their experiments use random vectors. This runs the same equations, unchanged, on I-JEPA
          embeddings of four real photographs.
        </p>
        <div className="actions rise" style={d(3)}>
          <a className="btn btn-primary" href="#results">
            See the results
          </a>
          <a className="btn btn-text" href={REPO}>
            View the code <span aria-hidden="true">→</span>
          </a>
        </div>
        <div className="rise" style={d(4)}>
          <Outcome />
        </div>
      </div>
    </section>
  );
}

function ModelChapter() {
  return (
    <>
      <Chapter
        id="model"
        sub={
          <div className="grid term-row">
            {FORCES.map((f) => (
              <span key={f.key} className="label ink">
                {f.name}
              </span>
            ))}
          </div>
        }
      >
        Model
      </Chapter>
      <Section
        id="equation"
        label="Equation 2"
        num={1}
        headline="Four forces act on every memory."
        lead="Each stored memory has an activation. All four start at 1 and evolve under the paper's equation (2). Scroll to take the forces one at a time."
      >
        <div className="grid">
          <div className="indent stack">
            <Card label="Equation 2" meta="Ghosh, Jaiswal and Gupta, section 2.7" className="equation-card">
              <Tex block>{String.raw`\frac{da}{dt} = -Aa \;-\; \gamma L a \;-\; \delta\,(\mathbf{1}^\top a)\,\mathbf{1} \;+\; \eta\, S(t)\, F(a)`}</Tex>
            </Card>
            <div className="prose-cols" data-reveal>
              <p>
                The first three terms are minus the gradient of an energy,{" "}
                <Tex>{"E(a) = \\tfrac12 a^\\top A a + \\tfrac{\\gamma}{2} a^\\top L a + \\tfrac{\\delta}{2}(\\mathbf{1}^\\top a)^2"}</Tex>
                , so without replay the system only loses energy. Replay is an outside push and is not part of <Tex>{"E"}</Tex>.
              </p>
              <p>
                The code steps forward with Euler's method, <Tex>{"a \\leftarrow \\max(0,\\, a + h\\,\\dot a)"}</Tex> with h = 0.01,
                as in the paper's equation (5), plus a floor at zero. Retrieval is this project's addition: a query scores memory i
                as <Tex>{"\\cos(q, x_i)\\cdot a_i"}</Tex>.
              </p>
            </div>
          </div>
        </div>
      </Section>
      <ForcesScrolly />
    </>
  );
}

function MemoriesSection() {
  return (
    <>
      <Section
        id="memories"
        label="The memories"
        num={2}
        headline="Four photos, four memories."
        lead={
          <>
            A beagle, a cat, a person and a French bulldog from Unsplash. Each card shows the photo, the decay rate it was given,
            and its activation when the run ended. Only the person survives.
          </>
        }
      />
      <MemoryStack />
    </>
  );
}

function HowItRunsSection() {
  return (
    <Section
      id="how-it-runs"
      label="Architecture"
      num={3}
      headline="From photo to memory to result."
      lead="Encoding needs the 2.5 GB I-JEPA checkpoint once. After that the dynamics are a few hundred lines of PyTorch that run in seconds."
    >
      <div className="grid">
        <div className="indent stack">
          <div className="card flow-card" data-reveal>
            <div className="card-head">
              <div className="label">The flow</div>
              <div className="meta">One pass, then 500 steps of the equation</div>
            </div>
            <Flow />
          </div>
          <div data-reveal>
            <Pipeline />
          </div>
        </div>
      </div>
    </Section>
  );
}

function SingleRunSection() {
  return (
    <Section
      id="single-run"
      label="One run"
      num={4}
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
          <Card
            label="Figure 2 · Colab output"
            meta="‖a(t)‖ · energy, log scale · final activations"
            note={
              <>
                The notebook's own figure, unchanged. Its axis uses the labels from the run: <code>golden_retriever</code> is the
                beagle, <code>pug</code> the person and <code>cute_dog</code> the French bulldog. ‖a‖ falls from {startNorm.toFixed(1)}{" "}
                to a noisy plateau near 0.15 from about step 280.
              </>
            }
            numbers={
              <table>
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
                  {memories.map((m) => (
                    <tr key={m.label}>
                      <td>{m.name}</td>
                      <td className="num">{m.decay_rate.toFixed(2)}</td>
                      <td className="num">{m.final === 0 ? "0" : m.final.toFixed(8)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <div className="plate">
              <img
                src={figures.demo}
                alt="Colab figure: activation norm falling from 2 to about 0.15, energy falling from about 3 to 0.015, and a bar chart where only the third memory is above zero."
              />
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function RetrievalSection() {
  return (
    <Section
      id="retrieval"
      label="Retrieval"
      num={5}
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
            The beagle's cosine similarity to the query is 1, but its activation is 0, so it scores 0. Once activations collapse,
            only the surviving memory can score above zero, whatever the query. Ranks 2 and 3 are tied at 0, so their order means
            nothing. The top score also gives one similarity for free: beagle and person embeddings have a cosine similarity of{" "}
            {top.score.toFixed(4)} / {run.final_activations[top.label].toFixed(4)} ≈ {impliedCosine.toFixed(2)}.
          </p>
        </div>
      </div>
    </Section>
  );
}

function RegimesSection() {
  return (
    <Section
      id="regimes"
      label="Regimes"
      num={6}
      headline="Three regimes, ordered by replay."
      lead="Section 4 of the paper predicts collapse, a balanced state and runaway reinforcement as replay gets stronger. The same four memories were run under three settings of η and p, twice."
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
                    <th>Sweep</th>
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
                    <tr key={`b${r.name}`}>
                      <td>Final settings</td>
                      <td>{r.name}</td>
                      <td className="num">{r.eta}</td>
                      <td className="num">{r.replay_prob}</td>
                      <td className="num">{drive(r)}</td>
                      <td className="num">{r.final_norm}</td>
                      <td className="num">{r.final_energy}</td>
                    </tr>
                  ))}
                  {sweepA.map((r) => (
                    <tr key={`a${r.name}`}>
                      <td>First sweep</td>
                      <td>{r.name}</td>
                      <td className="num">{r.eta}</td>
                      <td className="num">{r.replay_prob}</td>
                      <td className="num">{drive(r)}</td>
                      <td className="num">≈ {r.approx_final_norm}</td>
                      <td className="num">not printed</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <DriveChart
              exact={sweepB.map((r) => ({ name: r.name, drive: drive(r), norm: r.final_norm }))}
              approx={sweepA.map((r) => ({ name: r.name, drive: drive(r), norm: r.approx_final_norm }))}
              start={startNorm}
            />
          </Card>
          <Card
            label="Figure 4 · Final settings"
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
          <Card label="Figure 5 · First sweep" meta="Colab output" note="Slightly different settings. No numbers were printed, so its points in Figure 3 are read off this figure.">
            <div className="plate">
              <img
                src={figures.sweepA}
                alt="Colab figure: Collapse decays to zero, Balanced settles near 0.6, Runaway climbs to about 4.7."
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

function TheorySection() {
  const peaks: Record<string, string> = { Collapse: "≈ 2.0 (start)", Balanced: "≈ 2.0 (start)", Runaway: "≈ 6.1" };
  return (
    <Section
      id="theory"
      label="Theory"
      num={7}
      headline="Inside the paper's bounds."
      lead={
        <>
          Section 3.2 proves no trajectory leaves <Tex>{"\\|a\\| \\le \\max(\\|a(0)\\|,\\ \\eta / \\lambda_{\\min}(R))"}</Tex>, with{" "}
          <Tex>{"R = A + \\gamma L + \\delta\\mathbf{1}\\mathbf{1}^\\top"}</Tex>. Since <Tex>{"L"}</Tex> and{" "}
          <Tex>{"\\mathbf{1}\\mathbf{1}^\\top"}</Tex> are positive semidefinite, <Tex>{"\\lambda_{\\min}(R) \\ge \\min_i \\alpha_i"}</Tex> ={" "}
          {minDecay.toFixed(3)}, which gives a bound from the recorded decay rates alone.
        </>
      }
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="Boundedness · section 3.2" meta="Largest ‖a‖ read off Figure 4">
            <table>
              <thead>
                <tr>
                  <th>Regime</th>
                  <th className="num">
                    <N>η</N>
                  </th>
                  <th className="num">
                    Bound on <N>‖a‖</N>
                  </th>
                  <th className="num">
                    Largest <N>‖a‖</N> seen
                  </th>
                </tr>
              </thead>
              <tbody>
                {sweepB.map((r) => (
                  <tr key={r.name}>
                    <td>{r.name}</td>
                    <td className="num">{r.eta}</td>
                    <td className="num big">{bound(r.eta).toFixed(2)}</td>
                    <td className="num big">{peaks[r.name]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <div className="prose-cols" data-reveal>
            <div>
              <div className="label" style={{ marginBottom: 12 }}>
                Boundedness
              </div>
              <p>
                Every run stays inside its bound. Runaway is replay-dominated, not divergent: it levels off near 6, well under{" "}
                {bound(sweepB[2].eta).toFixed(1)}. Section 4.3 says strong replay can make an equilibrium unstable; section 3.2 says
                trajectories stay bounded regardless. This run fits the second.
              </p>
            </div>
            <div>
              <div className="label" style={{ marginBottom: 12 }}>
                Energy · sections 2.4 and 3.1
              </div>
              <p>
                Energy only has to fall with replay off, and the paper's own remark says replay need not preserve that. With replay
                on, the single run's energy falls about 200 times then wobbles, and runaway ends higher ({sweepB[2].final_energy.toFixed(1)}
                ) than it started (about {startEnergy.toFixed(2)}). Both fit the theory.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

function PaperSection() {
  return (
    <Section
      id="paper"
      label="The paper"
      num={8}
      headline="Same equations, real embeddings."
      lead="The paper proves its results for any embeddings, then illustrates them with synthetic data. Everything here that isn't about the embeddings is the same."
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="Setup" meta="Paper section 5 · this project's Colab run">
            <div className="scroll">
              <table className="compare">
                <thead>
                  <tr>
                    <th />
                    <th>Paper</th>
                    <th>This project</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Embeddings</td>
                    <td data-label="Paper">Random draws from a standard Gaussian</td>
                    <td data-label="This project">I-JEPA ViT-H/14 features of real photos, averaged over image patches</td>
                  </tr>
                  <tr>
                    <td>Dimension d</td>
                    <td data-label="Paper">64</td>
                    <td data-label="This project">1,280</td>
                  </tr>
                  <tr>
                    <td>Memories N</td>
                    <td data-label="Paper">200</td>
                    <td data-label="This project">{recorded.memories.length}</td>
                  </tr>
                  <tr>
                    <td>What the graph encodes</td>
                    <td data-label="Paper">
                      Nothing. Cosine similarity between independent random vectors centres on 0 with a spread of about 1/√64 ≈ 0.13,
                      so roughly half the edges are cut to zero.
                    </td>
                    <td data-label="This project">
                      How alike I-JEPA finds the photos. The run pins down one pair: beagle and person, about {impliedCosine.toFixed(2)}.
                    </td>
                  </tr>
                  <tr>
                    <td>Parameters</td>
                    <td colSpan={2}>
                      Same in both: γ = {recorded.base_params.gamma}, δ = {recorded.base_params.delta}, β = {recorded.base_params.beta}, h ={" "}
                      {recorded.base_params.h}, α ~ U[0.5, 1.5], S(t) ~ Bernoulli(p)
                    </td>
                  </tr>
                  <tr>
                    <td>Run length and seeds</td>
                    <td data-label="Paper">3,000 steps, averaged over 10 seeds</td>
                    <td data-label="This project">
                      {recorded.steps} steps, one seed ({recorded.seed})
                    </td>
                  </tr>
                  <tr>
                    <td>Regimes</td>
                    <td data-label="Paper">Collapse, balanced and runaway (Figures 1 and 3)</td>
                    <td data-label="This project">The same three, from three settings of η and p</td>
                  </tr>
                  <tr>
                    <td>Energy</td>
                    <td data-label="Paper">With replay off, falls steadily (Figure 4)</td>
                    <td data-label="This project">
                      With replay on, falls from about {startEnergy.toFixed(2)} to {run.final_energy.toFixed(3)}, then wobbles as replay adds
                      energy back
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
      <hr className="rule" />
      <div className="grid" style={{ marginTop: 40 }} data-reveal>
        <span className="label" style={{ gridColumn: "1 / 2", paddingTop: "0.5em" }}>
          Section 6.6
        </span>
        <blockquote className="quote indent">
          <p>
            “The experiments presented in this work focus on synthetic memory patterns. Applying the framework to realistic
            embeddings derived from language models or multimodal agents would provide a more direct evaluation of its practical
            relevance.”
          </p>
          <cite className="muted">
            Ghosh, Jaiswal and Gupta, Limitations and Future Directions. This project is a small first step at that: one image
            model, four photos, one seed.
          </cite>
        </blockquote>
      </div>
    </Section>
  );
}

function FindingsSection() {
  return (
    <Section id="findings" label="Findings" num={9} headline="What it shows, and what it doesn't.">
      <div className="grid">
        <div className="indent two-col">
          <div data-reveal>
            <div className="label">Shows</div>
            <ul className="points">
              <li>The paper's dynamics run unchanged on real 1,280-dimensional I-JEPA embeddings.</li>
              <li>They produce the collapse, balanced and runaway regimes the paper predicts.</li>
              <li>Every trajectory stays inside the paper's boundedness bound.</li>
              <li>In this run, replay concentrated on a single memory, and not the one that decays slowest.</li>
              <li>Retrieval weighted by activation returns whatever survived, even over an exact match.</li>
            </ul>
          </div>
          <div data-reveal style={d(1)}>
            <div className="label">Doesn't show</div>
            <ul className="points">
              <li>
                Whether similar memories help each other survive. The photos were a beagle, a cat, a person and a French bulldog, and
                the run didn't record how similar they are to each other.
              </li>
              <li>Anything statistical. One seed and four memories, where the paper uses 10 seeds and 200 memories.</li>
              <li>
                Why replay chose the person photo. The run didn't record the replay scores, the embedding lengths or the similarity
                between memories, so the reason can't be read off these results.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
}

function ReproduceSection() {
  return (
    <Section
      id="reproduce"
      label="Reproduce"
      num={10}
      headline="Run it yourself."
      lead={
        <>
          Open <code>notebooks/metacognition.ipynb</code> in Colab, or run the script, which repeats the notebook's cells in the order
          they ran with the same seed.
        </>
      }
    >
      <div className="grid">
        <div className="indent-narrow stack">
          <pre data-reveal>
            <code>
              {"pip install -r requirements.txt\n"}
              {"python run_experiments.py\n"}
              <span className="comment">{"# first run downloads I-JEPA (about 2.5 GB); later runs:\n"}</span>
              {"python run_experiments.py --embeddings outputs/embeddings.pt"}
            </code>
          </pre>
          <p className="muted" data-reveal>
            The numbers on this page are read straight from <code>results/metrics.json</code>. The notebook's printed output is copied
            word for word in <code>results/notebook_output.txt</code>.
          </p>
        </div>
      </div>
    </Section>
  );
}

/** Lines that come into focus one after another as the statement scrolls up. */
function Statement() {
  const [ref, p] = useEnterProgress<HTMLParagraphElement>(0.3);
  const lines = [
    "Random vectors showed the equations hold.",
    "Real embeddings show they run on perception.",
    "Next: does meaning change which memory survives?",
  ];
  return (
    <p className="statement" ref={ref}>
      {lines.map((l, i) => {
        const t = Math.min(1, Math.max(0, p * 3.2 - i * 0.9));
        return (
          <span
            key={l}
            className={i === lines.length - 1 ? "final" : undefined}
            style={{ filter: `blur(${(1 - t) * 10}px)`, opacity: 0.25 + 0.75 * t }}
          >
            {l}
          </span>
        );
      })}
    </p>
  );
}

function DarkClose() {
  return (
    <footer className="closing" data-dark>
      <div className="wrap">

        <Statement />
        <p className="sub">
          Everything on this page comes from one recorded Colab run: the printed numbers in metrics.json and the notebook's own
          figures.
        </p>

        <div className="footer-cols">
          <div>
            <div className="label">This project</div>
            <ul>
              <li>
                <a href="#model">The model</a>
              </li>
              <li>
                <a href="#how-it-runs">How it runs</a>
              </li>
              <li>
                <a href="#results">Results</a>
              </li>
              <li>
                <a href="#paper">The paper</a>
              </li>
              <li>
                <a href="#reproduce">Reproduce</a>
              </li>
            </ul>
          </div>
          <div>
            <div className="label">Source</div>
            <ul>
              <li>
                <a href={REPO}>GitHub repository</a>
              </li>
              <li>
                <a href={file("notebooks/metacognition.ipynb")}>Colab notebook</a>
              </li>
              <li>
                <a href={file("results/metrics.json")}>metrics.json</a>
              </li>
              <li>
                <a href={file("results/notebook_output.txt")}>Printed output</a>
              </li>
            </ul>
          </div>
          <div>
            <div className="label">Paper</div>
            <p>
              Priyam Ghosh, Krish Jaiswal, Sauhard Gupta. The Dynamics of Memory: Replay, Diffusion, and Attractor Formation.
              MetaCognition Labs, preprint under review. Encoder: I-JEPA (Assran et al., CVPR 2023). Photos: Unsplash License.
            </p>
          </div>
          <div>
            <div className="label">Built by</div>
            <p className="author">{AUTHOR}</p>
            <ul>
              <li>
                <a href={AUTHOR_GITHUB}>github.com/PavanSuryaChintada</a>
              </li>
            </ul>
            <a className="btn btn-light footer-btn" href={REPO}>
              View the code
            </a>
          </div>
        </div>
        <div className="footer-base">
          <span>© 2026 {AUTHOR}</span>
          <span className="footer-mark">
            <Mark size={18} /> Memory Dynamics
          </span>
        </div>
      </div>
    </footer>
  );
}
