import { fixes, forgetting, normalised, paperScale, paperTuned, ten } from "../data/results";
import { lateShareRange, normalisedFinals, unnormalisedFinals, winnerShare } from "../data/derived";
import { MiniBars } from "./Memories";
import { SurvivorGrid } from "./SurvivorGrid";
import { RetentionBars } from "../pages/BeyondPaper";
import { Tex } from "./Tex";
import { d } from "./ui";

const SCALE = 0.15; // shared bar scale, so the two cards compare directly

/** Run C's highest final activation as a share of the 0.001 survival line. */
export function ThresholdMeter() {
  const v = paperScale.max_activation;
  const line = paperScale.setup.alive_threshold;
  return (
    <div className="threshold-meter" role="img" aria-label={`Highest activation ${v}, ${Math.round((v / line) * 100)}% of the ${line} line`}>
      <div className="tm-track">
        <span className="tm-fill" style={{ width: `${(v / line) * 100}%` }} />
      </div>
      <div className="tm-labels">
        <span>
          highest <b>{v.toFixed(6)}</b>
        </span>
        <span>{line} line</span>
      </div>
    </div>
  );
}

/** The runs, side by side. Each card opens that run's page. */
export function RunCards() {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return (
    <div className="run-cards home-runs" id="runs">
      <a className="run-card rise" style={d(4)} href="#/unnormalised">
        <div className="run-card-head">
          <span className="label">Run A</span>
          <span className="tag ok">As the paper writes it</span>
        </div>
        <h2 className="run-title">Unnormalised replay</h2>
        <div className="run-formula">
          <Tex>{"F(a) = \\mathrm{softmax}(\\beta\\, X^\\top X a)"}</Tex>
        </div>
        <p className="run-desc">
          Replay scores use the raw I-JEPA embeddings. Their dot products are large, so the softmax puts all of its weight on one
          memory, and that memory is the only one left.
        </p>
        <MiniBars finals={unnormalisedFinals} scale={SCALE} />
        <dl className="run-stats">
          <div>
            <dt>Memories left</dt>
            <dd>
              1 of 4{" "}
              <span className="dd-sub">
                {ten.survivors_above_0_001.unnormalised} of {ten.memories.length} with ten photos
              </span>
            </dd>
          </div>
          <div>
            <dt>Replay to one memory</dt>
            <dd>{pct(winnerShare)}</dd>
          </div>
        </dl>
        <span className="run-open">
          Open run A <span aria-hidden="true">→</span>
        </span>
      </a>

      <a className="run-card rise" style={d(5)} href="#/normalised">
        <div className="run-card-head">
          <span className="label">Run B</span>
          <span className="tag new">One change</span>
        </div>
        <h2 className="run-title">Normalised replay</h2>
        <div className="run-formula">
          <Tex>{"F(a) = \\mathrm{softmax}(\\beta\\, \\hat X^\\top \\hat X a)"}</Tex>
        </div>
        <p className="run-desc">
          The same run, with each embedding scaled to length 1 inside the replay operator only. Replay spreads across all four
          memories, and every one of them stays, in order of decay rate. Repeated with ten photos: all ten stay.
        </p>
        <MiniBars finals={normalisedFinals} scale={SCALE} />
        <dl className="run-stats">
          <div>
            <dt>Memories left</dt>
            <dd>
              {normalised.survivors_above_0_001.normalised} of 4{" "}
              <span className="dd-sub">
                {ten.survivors_above_0_001.normalised} of {ten.memories.length} with ten photos
              </span>
            </dd>
          </div>
          <div>
            <dt>Replay share each</dt>
            <dd>
              {pct(lateShareRange[0])} to {pct(lateShareRange[1])}
            </dd>
          </div>
        </dl>
        <span className="run-open">
          Open run B <span aria-hidden="true">→</span>
        </span>
      </a>
      <a className="run-card rise" style={d(6)} href="#/paper">
        <div className="run-card-head">
          <span className="label">Run C</span>
          <span className="tag note">N = {paperScale.setup.n}</span>
        </div>
        <h2 className="run-title">Paper scale</h2>
        <div className="run-formula">
          <Tex>{String.raw`N = ${paperScale.setup.n},\quad T = ${paperScale.setup.steps}`}</Tex>
        </div>
        <p className="run-desc">
          Run B's replay at the paper's size: {paperScale.setup.n} CIFAR-100 images through I-JEPA for {paperScale.setup.steps.toLocaleString("en")}{" "}
          steps. With the paper's δ none stays above {paperScale.setup.alive_threshold}; a sweep of γ and δ finds where they do.
        </p>
        <SurvivorGrid />
        <dl className="run-stats">
          <div>
            <dt>Memories left</dt>
            <dd>
              {paperScale.survivors} → {paperTuned.best_printed.alive}{" "}
              <span className="dd-sub">
                δ = {paperScale.setup.delta} → {paperTuned.best_printed.delta}
              </span>
            </dd>
          </div>
          <div>
            <dt>Order by decay</dt>
            <dd>
              ρ ≈ 0.99 <span className="dd-sub">in every run</span>
            </dd>
          </div>
        </dl>
        <span className="run-open">
          Open run C <span aria-hidden="true">→</span>
        </span>
      </a>
      <a className="run-card rise" style={d(7)} href="#/beyond">
        <div className="run-card-head">
          <span className="label">Beyond the paper</span>
          <span className="tag note">2 tests</span>
        </div>
        <h2 className="run-title">New memories mid-run</h2>
        <div className="run-formula">
          <Tex>{String.raw`a_{\text{new}}(t_{${forgetting.setup.inject_at}}) = 1 \;\text{or}\; \bar a`}</Tex>
        </div>
        <p className="run-desc">
          Five memories at step 0, five more added at step {forgetting.setup.inject_at}, with run B's replay. Test 1 adds them at full
          strength; test 2 compares gentler ways of adding them.
        </p>
        <RetentionBars />
        <dl className="run-stats">
          <div>
            <dt>Old activation kept</dt>
            <dd>
              {fixes.retention_pct_printed.raw}% → {fixes.retention_pct_printed.mean}%{" "}
              <span className="dd-sub">new ones start at 1 → at the mean</span>
            </dd>
          </div>
          <div>
            <dt>Old memories left</dt>
            <dd>
              {fixes.survivors.raw.old} of 5 <span className="dd-sub">with a photo in every slot</span>
            </dd>
          </div>
        </dl>
        <span className="run-open">
          Open the tests <span aria-hidden="true">→</span>
        </span>
      </a>
      <p className="run-cards-note muted small">
        Runs A and B: bars show each memory's activation at step 500 on the same scale, 0 to {SCALE}, with the same photos, decay
        rates, parameters and seed. Run C: memories above 0.001 at step {paperScale.setup.steps.toLocaleString("en")} for each γ (rows) and δ
        (columns); the dashed cell is the paper's. Beyond the paper: the share of the old memories' activation kept from step 250 to
        step 500, for each way of adding new ones.
      </p>
    </div>
  );
}
