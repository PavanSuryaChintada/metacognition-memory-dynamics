import { figures, memories, nameOf, recorded, REPO } from "./data/results";
import { Card, Chapter, Header, Section } from "./components/Layout";
import { HeroArt } from "./components/HeroArt";
import { MemoryCards, Outcome } from "./components/Memories";
import { Pipeline } from "./components/Pipeline";
import { DriveChart } from "./components/DriveChart";
import { Tex } from "./components/Tex";

const run = recorded.single_run;
const sweepA = recorded.regimes.sweep_a.runs;
const sweepB = recorded.regimes.sweep_b.runs;
const decays = recorded.memories.map((m) => m.decay_rate);
const minDecay = Math.min(...decays);
const slowest = recorded.memories[decays.indexOf(minDecay)];
const survivor = memories.find((m) => m.final > 0);
const startEnergy = 0.5 * decays.reduce((s, d) => s + d, 0) + 0.5 * recorded.base_params.delta * decays.length ** 2;
const top = run.retrieval.top[0];
const impliedCosine = top.score / run.final_activations[top.label];
const startNorm = Math.sqrt(recorded.memories.length);
const drive = (r: { eta: number; replay_prob: number }) => +(r.eta * r.replay_prob).toFixed(4);
const bound = (eta: number) => Math.max(startNorm, eta / minDecay);
const file = (path: string) => `${REPO}/blob/main/${path}`;

export function App() {
  return (
    <>
      <Header />
      <main>
        <Hero />
        <Chapter id="method">Method</Chapter>
        <PaperSection />
        <ModelSection />
        <ArchitectureSection />
        <MemoriesSection />
        <Chapter id="results">Results</Chapter>
        <SingleRunSection />
        <RetrievalSection />
        <RegimesSection />
        <TheorySection />
        <FindingsSection />
        <ReproduceSection />
      </main>
      <Closing />
    </>
  );
}

function Hero() {
  return (
    <section className="hero" id="top">
      <HeroArt />
      <div className="wrap hero-inner">
        <p className="label">A replication on real embeddings</p>
        <h1 className="display">Memory dynamics on real perception.</h1>
        <p className="hero-sub">
          Ghosh, Jaiswal and Gupta model memory as activations that decay, spread across a similarity graph, and get
          reinforced by replay. Their experiments use random vectors. This runs the same equations, unchanged, on I-JEPA
          embeddings of four real photographs.
        </p>
        <div className="actions">
          <a className="btn btn-primary" href="#results">
            See the results
          </a>
          <a className="btn btn-text" href={REPO}>
            View on GitHub →
          </a>
        </div>
        <Outcome />
      </div>
    </section>
  );
}

function PaperSection() {
  return (
    <Section
      id="paper"
      label="The paper"
      num={1}
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
                      Nothing. Cosine similarity between independent random vectors centres on 0 with a spread of about
                      1/√64 ≈ 0.13, so roughly half the edges are cut to zero.
                    </td>
                    <td data-label="This project">
                      How alike I-JEPA finds the photos. The run pins down one pair: beagle and person, about{" "}
                      {impliedCosine.toFixed(2)}.
                    </td>
                  </tr>
                  <tr>
                    <td>Parameters</td>
                    <td colSpan={2}>
                      Same in both: γ = {recorded.base_params.gamma}, δ = {recorded.base_params.delta}, β ={" "}
                      {recorded.base_params.beta}, h = {recorded.base_params.h}, α ~ U[0.5, 1.5], S(t) ~ Bernoulli(p)
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
                      With replay on, falls from about {startEnergy.toFixed(2)} to {run.final_energy.toFixed(3)}, then
                      wobbles as replay adds energy back
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
      <hr className="rule" />
      <div className="grid" style={{ marginTop: 40 }}>
        <span className="label" style={{ gridColumn: "1 / 2", paddingTop: "0.5em" }}>
          Section 6.6
        </span>
        <blockquote className="quote indent">
          <p>
            “The experiments presented in this work focus on synthetic memory patterns. Applying the framework to realistic
            embeddings derived from language models or multimodal agents would provide a more direct evaluation of its
            practical relevance.”
          </p>
          <cite className="muted">
            Ghosh, Jaiswal and Gupta, Limitations and Future Directions. This project is a small first step at that: one
            image model, four photos, one seed.
          </cite>
        </blockquote>
      </div>
    </Section>
  );
}

function ModelSection() {
  return (
    <Section
      id="model"
      label="The model"
      num={2}
      headline="Four forces on every memory."
      lead="Each stored memory has an activation. All of them start at 1 and evolve under the paper's equation (2)."
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="Equation 2" meta="Ghosh, Jaiswal and Gupta, section 2.7" className="equation-card">
            <Tex block>{String.raw`\frac{da}{dt} = -Aa \;-\; \gamma L a \;-\; \delta\,(\mathbf{1}^\top a)\,\mathbf{1} \;+\; \eta\, S(t)\, F(a)`}</Tex>
          </Card>
          <div className="terms">
            <div className="term">
              <div className="label">Decay</div>
              <div className="math">
                <Tex>{"-Aa"}</Tex>
              </div>
              <p>Each memory fades at its own rate α, drawn once from U[0.5, 1.5].</p>
            </div>
            <div className="term">
              <div className="label">Diffusion</div>
              <div className="math">
                <Tex>{"-\\gamma L a"}</Tex>
              </div>
              <p>Activation spreads between similar memories, along a graph whose edges are cosine similarity clipped at zero.</p>
            </div>
            <div className="term">
              <div className="label">Inhibition</div>
              <div className="math">
                <Tex>{"-\\delta(\\mathbf{1}^\\top a)\\mathbf{1}"}</Tex>
              </div>
              <p>The total activation pushes every memory down equally.</p>
            </div>
            <div className="term">
              <div className="label">Replay</div>
              <div className="math">
                <Tex>{"\\eta S(t) F(a)"}</Tex>
              </div>
              <p>
                When the coin flip lands (probability p), memories like the current state are reinforced through a softmax that
                sums to 1.
              </p>
            </div>
          </div>
          <div className="prose-cols" style={{ marginTop: 32 }}>
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
  );
}

function ArchitectureSection() {
  return (
    <Section
      id="architecture"
      label="Architecture"
      num={3}
      headline="How the code fits together."
      lead="Encoding needs the 2.5 GB I-JEPA checkpoint. The dynamics are a few hundred lines of PyTorch that run in seconds."
    >
      <div className="grid">
        <div className="indent">
          <Pipeline />
        </div>
      </div>
    </Section>
  );
}

function MemoriesSection() {
  return (
    <Section
      id="memories"
      label="The memories"
      num={4}
      headline="Four photos, two of them mislabelled."
      lead={
        <>
          The notebook's labels were written for an earlier set of images. In this run <code>golden_retriever</code> is a beagle
          and <code>pug</code> is a photo of a person. The Colab figures below keep the notebook's labels.
        </>
      }
    >
      <div className="grid">
        <div className="indent">
          <MemoryCards />
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
      num={5}
      headline="One memory survives."
      lead={`Replay strength η = ${run.eta}, replay probability p = ${run.replay_prob}, ${recorded.steps} steps, seed ${recorded.seed}.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Survivors">
              <div className="stat-value ember">
                1<span className="unit">of {recorded.memories.length}</span>
              </div>
              <div className="stat-sub">
                The {survivor?.name.toLowerCase()} photo keeps {survivor?.final.toFixed(4)}. The rest reach exactly 0.
              </div>
            </Card>
            <Card label="Slowest decay">
              <div className="stat-value">
                α = {minDecay.toFixed(2)}
              </div>
              <div className="stat-sub">
                The {nameOf(slowest.label).toLowerCase()}, and it was still forgotten. Replay picked the winner, not decay.
              </div>
            </Card>
            <Card label="Energy">
              <div className="stat-value">
                {startEnergy.toFixed(2)} <span className="unit">to</span> {run.final_energy.toFixed(4)}
              </div>
              <div className="stat-sub">About 200 times lower, then flat where replay adds back what decay removes.</div>
            </Card>
          </div>
          <Card
            label="Figure 2 · Colab output"
            meta="‖a(t)‖ · energy, log scale · final activations"
            note={
              <>
                The notebook's own figure, unchanged. Its labels are the notebook's: read <code>pug</code> as the person,{" "}
                <code>golden_retriever</code> as the beagle and <code>cute_dog</code> as the French bulldog. ‖a‖ falls from{" "}
                {startNorm.toFixed(1)} to a noisy plateau near 0.15 from about step 280.
              </>
            }
            numbers={
              <table>
                <thead>
                  <tr>
                    <th>Memory</th>
                    <th className="num">Decay rate <span className="nocase">α</span></th>
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
      num={6}
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
          <p className="muted">
            The beagle's cosine similarity to the query is 1, but its activation is 0, so it scores 0. Once activations collapse,
            only the surviving memory can score above zero, whatever the query. Ranks 2 and 3 are tied at 0, so their order
            means nothing. The top score also gives one similarity for free: beagle and person embeddings have a cosine
            similarity of {top.score.toFixed(4)} / {run.final_activations[top.label].toFixed(4)} ≈ {impliedCosine.toFixed(2)}.
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
      num={7}
      headline="Three regimes, ordered by replay."
      lead="Section 4 of the paper predicts collapse, a balanced state and runaway reinforcement as replay gets stronger. The same four memories were run under three settings of η and p, twice."
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label={<>Figure 3 · Final <span className="nocase">‖a‖</span> against replay drive</>}
            meta="Log scales · seed 42 · 500 steps"
            numbers={
              <table>
                <thead>
                  <tr>
                    <th>Sweep</th>
                    <th>Regime</th>
                    <th className="num"><span className="nocase">η</span></th>
                    <th className="num"><span className="nocase">p</span></th>
                    <th className="num"><span className="nocase">η·p</span></th>
                    <th className="num">Final <span className="nocase">‖a‖</span></th>
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
          <div className="stack">
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
          </div>
          <ul className="points">
            <li>
              <b>Collapse</b>, η·p = {drive(sweepB[0])}. Replay is too weak to matter; everything decays to ‖a‖ ={" "}
              {sweepB[0].final_norm}.
            </li>
            <li>
              <b>Balanced</b>, η·p = {drive(sweepB[1])}. ‖a‖ drops over the first 100 steps, then hovers between about 0.45 and
              0.65.
            </li>
            <li>
              <b>Runaway</b>, η·p = {drive(sweepB[2])}. ‖a‖ climbs from 2 to about 6 and levels off; energy ends at{" "}
              {sweepB[2].final_energy.toFixed(1)}, about {Math.round(sweepB[2].final_energy / startEnergy)} times where it
              started.
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
      num={8}
      headline="Inside the paper's bounds."
      lead={
        <>
          Section 3.2 proves no trajectory leaves <Tex>{"\\|a\\| \\le \\max(\\|a(0)\\|,\\ \\eta / \\lambda_{\\min}(R))"}</Tex>, with{" "}
          <Tex>{"R = A + \\gamma L + \\delta\\mathbf{1}\\mathbf{1}^\\top"}</Tex>. Since <Tex>{"L"}</Tex> and{" "}
          <Tex>{"\\mathbf{1}\\mathbf{1}^\\top"}</Tex> are positive semidefinite, <Tex>{"\\lambda_{\\min}(R) \\ge \\min_i \\alpha_i"}</Tex>{" "}
          = {minDecay.toFixed(3)}, which gives a bound from the recorded decay rates alone.
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
                  <th className="num"><span className="nocase">η</span></th>
                  <th className="num">Bound on <span className="nocase">‖a‖</span></th>
                  <th className="num">Largest <span className="nocase">‖a‖</span> seen</th>
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
          <div className="prose-cols">
            <div>
              <div className="label" style={{ marginBottom: 12 }}>
                Boundedness
              </div>
              <p>
                Every run stays inside its bound. Runaway is replay-dominated, not divergent: it levels off near 6, well under{" "}
                {bound(sweepB[2].eta).toFixed(1)}. Section 4.3 says strong replay can make an equilibrium unstable; section 3.2
                says trajectories stay bounded regardless. This run fits the second.
              </p>
            </div>
            <div>
              <div className="label" style={{ marginBottom: 12 }}>
                Energy · sections 2.4 and 3.1
              </div>
              <p>
                Energy only has to fall with replay off, and the paper's own remark says replay need not preserve that. With
                replay on, the single run's energy falls about 200 times then wobbles, and runaway ends higher (
                {sweepB[2].final_energy.toFixed(1)}) than it started (about {startEnergy.toFixed(2)}). Both fit the theory.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

function FindingsSection() {
  return (
    <Section id="findings" label="Findings" num={9} headline="What it shows, and what it doesn't.">
      <div className="grid">
        <div className="indent two-col">
          <div>
            <div className="label">Shows</div>
            <ul className="points">
              <li>The paper's dynamics run unchanged on real 1,280-dimensional I-JEPA embeddings.</li>
              <li>They produce the collapse, balanced and runaway regimes the paper predicts.</li>
              <li>Every trajectory stays inside the paper's boundedness bound.</li>
              <li>With these settings, replay concentrates on a single memory, and not the one that decays slowest.</li>
              <li>Retrieval weighted by activation returns whatever survived, even over an exact match.</li>
            </ul>
          </div>
          <div>
            <div className="label">Doesn't show</div>
            <ul className="points">
              <li>
                Whether similar memories help each other survive. The photos were a beagle, a cat, a person and a French
                bulldog, not the planned set of dogs, and the run didn't record how similar they are.
              </li>
              <li>Anything statistical. One seed and four memories, where the paper uses 10 seeds and 200 memories.</li>
              <li>
                Why replay chose the person photo. The replay score uses raw, unnormalised embeddings, so a vector's length counts
                as well as its direction, and the lengths weren't recorded.
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
          Open <code>notebooks/metacognition.ipynb</code> in Colab, or run the script, which repeats the notebook's cells in the
          order they ran with the same seed.
        </>
      }
    >
      <div className="grid">
        <div className="indent-narrow stack">
          <pre>
            <code>
              {"pip install -r requirements.txt\n"}
              {"python run_experiments.py\n"}
              <span className="comment">{"# first run downloads I-JEPA (about 2.5 GB); later runs:\n"}</span>
              {"python run_experiments.py --embeddings outputs/embeddings.pt"}
            </code>
          </pre>
          <p className="muted">
            The numbers on this page are read straight from <code>results/metrics.json</code>. The notebook's printed output is
            copied word for word in <code>results/notebook_output.txt</code>.
          </p>
        </div>
      </div>
    </Section>
  );
}

function Closing() {
  return (
    <footer className="closing">
      <div className="wrap">
        <p className="statement">
          <span>Random vectors showed the equations hold.</span>
          <span>Real embeddings show they run on perception.</span>
          <span>Next: does meaning change which memory survives?</span>
        </p>
        <p className="sub">
          Everything on this page comes from one recorded Colab run: the printed numbers in metrics.json and the notebook's own
          figures.
        </p>
        <div className="footer-cols">
          <div>
            <div className="label">On this page</div>
            <ul>
              <li>
                <a href="#paper">The paper</a>
              </li>
              <li>
                <a href="#model">The model</a>
              </li>
              <li>
                <a href="#results">Results</a>
              </li>
              <li>
                <a href="#theory">Theory</a>
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
              MetaCognition Labs, preprint under review.
            </p>
          </div>
          <div>
            <div className="label">Credits</div>
            <p>
              Encoder: I-JEPA (Assran et al., CVPR 2023),{" "}
              <a href="https://huggingface.co/facebook/ijepa_vith14_1k">facebook/ijepa_vith14_1k</a>. Photos from Unsplash, under
              the Unsplash License.
            </p>
          </div>
        </div>
        <div className="footer-base">
          <span>Built from the recorded Colab run</span>
          <span>
            Seed {recorded.seed} · {recorded.steps} steps
          </span>
        </div>
      </div>
    </footer>
  );
}
