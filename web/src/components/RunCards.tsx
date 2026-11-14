import { normalised } from "../data/results";
import { lateShareRange, normalisedFinals, unnormalisedFinals, winnerShare } from "../data/derived";
import { MiniBars } from "./Memories";
import { Tex } from "./Tex";
import { d } from "./ui";

const SCALE = 0.15; // shared bar scale, so the two cards compare directly

/** The two runs, side by side. Each card opens that run's page. */
export function RunCards() {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return (
    <div className="run-cards" id="runs">
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
            <dd>1 of 4</dd>
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
          memories, and every one of them stays, in order of decay rate.
        </p>
        <MiniBars finals={normalisedFinals} scale={SCALE} />
        <dl className="run-stats">
          <div>
            <dt>Memories left</dt>
            <dd>{normalised.survivors_above_0_001.normalised} of 4</dd>
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
      <p className="run-cards-note muted small">
        Bars show each memory's activation at step 500 on the same scale, 0 to {SCALE}. Same photos, decay rates, parameters and
        seed in both runs.
      </p>
    </div>
  );
}
