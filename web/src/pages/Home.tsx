import { recorded, REPO } from "../data/results";
import { Card, Chapter, Section } from "../components/Layout";
import { HeroArt } from "../components/HeroArt";
import { RunCards } from "../components/RunCards";
import { FORCES, ForcesScrolly } from "../components/Forces";
import { MemoryStack } from "../components/MemoryStack";
import { Flow } from "../components/Flow";
import { Pipeline } from "../components/Pipeline";
import { Tex } from "../components/Tex";
import { d } from "../components/ui";

/** The overview: what stays the same in every run. Results live on the run pages. */
export function Home() {
  return (
    <>
      <Hero />
      <ModelChapter />
      <MemoriesSection />
      <HowItRunsSection />
      <PaperSection />
      <ReproduceSection />
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
          Ghosh, Jaiswal and Gupta model memory as activations that decay, spread across a similarity graph, and get reinforced by
          replay. Their experiments use random vectors. Here the same equations run on I-JEPA embeddings of four real photographs,
          twice, with one difference in how replay scores the memories.
        </p>
        <div className="actions rise" style={d(3)}>
          <a className="btn btn-primary" href="#runs">
            Compare the two runs
          </a>
          <a className="btn btn-text" href={REPO}>
            View the code <span aria-hidden="true">→</span>
          </a>
        </div>
        <RunCards />
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
                as in the paper's equation (5), plus a floor at zero. The two runs differ only in the replay scores inside{" "}
                <Tex>{"F(a)"}</Tex>: raw embeddings in run A, embeddings scaled to length 1 in run B.
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
        lead="A beagle, a cat, a person and a French bulldog from Unsplash. Each is encoded once by I-JEPA and stored with its own decay rate, the same in both runs."
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

function PaperSection() {
  return (
    <Section
      id="paper"
      label="The paper"
      num={4}
      headline="Same equations, real embeddings."
      lead="The paper proves its results for any embeddings, then illustrates them with synthetic data. Everything here that isn't about the embeddings is the same."
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="Setup" meta="Paper section 5 · these runs">
            <div className="scroll">
              <table className="compare">
                <thead>
                  <tr>
                    <th />
                    <th>Paper</th>
                    <th>These runs</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Embeddings</td>
                    <td data-label="Paper">Random draws from a standard Gaussian</td>
                    <td data-label="These runs">I-JEPA ViT-H/14 features of real photos, averaged over image patches</td>
                  </tr>
                  <tr>
                    <td>Dimension d</td>
                    <td data-label="Paper">64</td>
                    <td data-label="These runs">1,280</td>
                  </tr>
                  <tr>
                    <td>Memories N</td>
                    <td data-label="Paper">200</td>
                    <td data-label="These runs">{recorded.memories.length}</td>
                  </tr>
                  <tr>
                    <td>What the graph encodes</td>
                    <td data-label="Paper">
                      Nothing. Cosine similarity between independent random vectors centres on 0 with a spread of about 1/√64 ≈ 0.13,
                      so roughly half the edges are cut to zero.
                    </td>
                    <td data-label="These runs">How alike I-JEPA finds the photos. Measured: every pair between 0.30 and 0.47, so no edge is zero.</td>
                  </tr>
                  <tr>
                    <td>Replay scores</td>
                    <td data-label="Paper">
                      <Tex>{"F(a) = \\mathrm{softmax}(\\beta X^\\top X a)"}</Tex>, raw embeddings
                    </td>
                    <td data-label="These runs">Run A as the paper writes it; run B with each embedding scaled to length 1</td>
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
                    <td data-label="These runs">
                      {recorded.steps} steps, one seed ({recorded.seed})
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
            “The experiments presented in this work focus on synthetic memory patterns. Applying the framework to realistic embeddings
            derived from language models or multimodal agents would provide a more direct evaluation of its practical relevance.”
          </p>
          <cite className="muted">
            Ghosh, Jaiswal and Gupta, Limitations and Future Directions. These runs are a small first step at that: one image model,
            four photos, one seed.
          </cite>
        </blockquote>
      </div>
    </Section>
  );
}

function ReproduceSection() {
  return (
    <Section
      id="reproduce"
      label="Reproduce"
      num={5}
      headline="Run it yourself."
      lead="Three recorded Colab runs, all with the same photos, decay rates and seed. Each one's printed output is kept word for word in results/."
    >
      <div className="grid">
        <div className="indent stack">
          <div className="card" data-reveal>
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Code</th>
                  <th>Recorded output</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>The notebook (run A)</td>
                  <td>
                    <code>notebooks/metacognition.ipynb</code>
                  </td>
                  <td>
                    <code>results/notebook_output.txt</code>
                  </td>
                </tr>
                <tr>
                  <td>Replay tracking (run A)</td>
                  <td>
                    <code>notebooks/replay_tracking.py</code>
                  </td>
                  <td>
                    <code>results/replay_tracking_output.txt</code>
                  </td>
                </tr>
                <tr>
                  <td>Raw vs normalised (run B)</td>
                  <td>
                    <code>notebooks/normalised_replay.py</code>
                  </td>
                  <td>
                    <code>results/normalised_output.txt</code>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <pre data-reveal>
            <code>
              {"pip install -r requirements.txt\n"}
              {"python run_experiments.py\n"}
              <span className="comment">{"# first run downloads I-JEPA (about 2.5 GB); later runs:\n"}</span>
              {"python run_experiments.py --embeddings outputs/embeddings.pt"}
            </code>
          </pre>
        </div>
      </div>
    </Section>
  );
}
