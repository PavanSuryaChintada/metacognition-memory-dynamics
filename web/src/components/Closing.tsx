import { AUTHOR, AUTHOR_GITHUB, NOTEBOOKS, REPO } from "../data/results";
import { useEnterProgress } from "../hooks";
import { Mark } from "./Layout";

/** Lines that come into focus one after another as the statement scrolls up. */
function Statement() {
  const [ref, p] = useEnterProgress<HTMLParagraphElement>(0.3);
  const lines = [
    "Same memories, same seed, same equations.",
    "Raw replay scores: one memory takes every replay.",
    "Unit-length scores: every memory stays, four of four and ten of ten.",
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

export function Closing() {
  return (
    <footer className="closing" data-dark>
      <div className="wrap">
        <Statement />
        <p className="sub">
          Everything here comes from eleven recorded Colab notebooks: the original notebook, the replay-tracking run, the
          raw-versus-normalised comparisons with 4 and 10 memories, two at the paper's scale, two adding memories mid-run, and three
          of Bayesian confidence.
        </p>

        <div className="footer-cols">
          <div>
            <div className="label">This project</div>
            <ul>
              <li>
                <a href="#runs">All runs</a>
              </li>
              <li>
                <a href="#/unnormalised">Run A, unnormalised</a>
              </li>
              <li>
                <a href="#/normalised">Run B, normalised</a>
              </li>
              <li>
                <a href="#/paper">Run C, paper scale</a>
              </li>
              <li>
                <a href="#/beyond">Beyond the paper, new memories</a>
              </li>
              <li>
                <a href="#/bayes">Beyond the paper, confidence</a>
              </li>
              <li>
                <a href="#model">The model</a>
              </li>
              <li>
                <a href="#how-it-runs">How it runs</a>
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
                <a href={NOTEBOOKS.original}>Original notebook</a>
              </li>
              <li>
                <a href={NOTEBOOKS.unnormalised4}>Run A, replay tracking</a>
              </li>
              <li>
                <a href={NOTEBOOKS.normalised4}>Run B, 4 memories</a>
              </li>
              <li>
                <a href={NOTEBOOKS.normalised10}>Run B, 10 memories</a>
              </li>
              <li>
                <a href={NOTEBOOKS.forgetting}>Beyond the paper, test 1</a>
              </li>
              <li>
                <a href={NOTEBOOKS.forgettingFixes}>Beyond the paper, test 2</a>
              </li>
              <li>
                <a href={NOTEBOOKS.paperScale}>Run C, paper scale</a>
              </li>
              <li>
                <a href={NOTEBOOKS.paperTuned}>Run C, γ and δ sweep</a>
              </li>
              <li>
                <a href={NOTEBOOKS.bayesFour}>Confidence, test 1</a>
              </li>
              <li>
                <a href={NOTEBOOKS.bayes200}>Confidence, test 2</a>
              </li>
              <li>
                <a href={NOTEBOOKS.bayesRelative}>Confidence, test 3</a>
              </li>
              <li>
                <a href={`${REPO}/tree/main/notebooks/colab`}>All Colab notebooks</a>
              </li>
              <li>
                <a href={`${REPO}/tree/main/results`}>Recorded results</a>
              </li>
            </ul>
          </div>
          <div>
            <div className="label">Paper</div>
            <p>
              Priyam Ghosh, Krish Jaiswal, Sauhard Gupta. The Dynamics of Memory: Replay, Diffusion, and Attractor Formation. MetaCognition
              Labs, preprint under review. Encoder: I-JEPA (Assran et al., CVPR 2023). Photos: Unsplash License.
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
