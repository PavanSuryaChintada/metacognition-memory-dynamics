import type { CSSProperties } from "react";
import { figures, nameOf, recorded, replay } from "../data/results";
import { Card, Section } from "./Layout";

const N = ({ children }: { children: string }) => <span className="nocase">{children}</span>;
const FORCE_KEYS = ["decay", "diffusion", "inhibition", "replay", "net"] as const;
const signed = (v: number) => (v > 0 ? `+${v.toFixed(5)}` : v === 0 ? "0" : v.toFixed(5));

/** The second Colab run: replay weights, measured similarity and the force breakdown. */
export function ReplaySection({ num }: { num: number }) {
  const labels = replay.labels;
  const winner = labels.find((l) => replay.final_activations[l] > 0)!;
  const share = replay.replay_weight_mean.first_10pct_of_events[winner];
  const lastShare = replay.replay_weight_mean.last_10pct_of_events[winner];
  const recordedFinal = recorded.single_run.final_activations[winner];
  const sim = replay.cosine_similarity;
  const offDiag = sim.flatMap((row, i) => row.filter((_, j) => j !== i));
  const [lo, hi] = [Math.min(...offDiag), Math.max(...offDiag)];
  const rowSum = (i: number) => sim[i].reduce((s, v, j) => (j === i ? s : s + v), 0);
  const fb = labels.indexOf("french_bulldog");
  const person = labels.indexOf("person");
  const beagle = labels.indexOf("beagle");

  return (
    <Section
      id="replay"
      label="Replay tracking"
      num={num}
      headline="Replay went to one memory, every time."
      lead={`A second Colab run repeated the same setup and logged the softmax replay weights F(a) at every step. It reproduces the first run: the ${nameOf(winner).toLowerCase()} ends at ${replay.final_activations[winner]}, against ${recordedFinal} in the notebook.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            <Card label="Replay events" meta={`p = ${replay.setup.replay_prob}`}>
              <div className="stat-value">
                {replay.replay_events}
                <span className="unit">of {replay.setup.steps} steps</span>
              </div>
              <div className="stat-sub">Steps where the replay coin flip fired.</div>
            </Card>
            <Card label={`${nameOf(winner)}'s replay share`} meta="F(a), 4 decimals" delay={1}>
              <div className="stat-value ember">{share.toFixed(4)}</div>
              <div className="stat-sub">
                Mean over the first 10% of replay events, and {lastShare.toFixed(4)} over the last 10%. The other three get 0.0000.
              </div>
            </Card>
            <Card label="Reproduces the first run" meta="Final activation" delay={2}>
              <div className="stat-value">{replay.final_activations[winner]}</div>
              <div className="stat-sub">
                The {nameOf(winner).toLowerCase()}'s value at step {replay.setup.steps}. The notebook printed {recordedFinal}.
              </div>
            </Card>
          </div>

          <Card
            label="Cosine similarity · measured"
            meta="Full 1,280-number embeddings"
            note={
              <>
                Every pair is positive, from {lo.toFixed(2)} to {hi.toFixed(2)}, so no edge of the graph is zero. The French
                bulldog's closest match is the person ({sim[fb][person].toFixed(2)}), not the beagle ({sim[fb][beagle].toFixed(2)}).
                The beagle–person value, {sim[beagle][person].toFixed(4)}, is the 0.30 the first run's retrieval score implied.
              </>
            }
          >
            <div className="scroll">
              <table className="sim-table">
                <thead>
                  <tr>
                    <th />
                    {labels.map((l) => (
                      <th key={l} className="num">
                        {nameOf(l)}
                      </th>
                    ))}
                    <th className="num">Sum of the other three</th>
                  </tr>
                </thead>
                <tbody>
                  {labels.map((l, i) => (
                    <tr key={l}>
                      <td>{nameOf(l)}</td>
                      {sim[i].map((v, j) => (
                        <td
                          key={j}
                          className={`num${i === j ? " diag" : ""}`}
                          style={i === j ? undefined : ({ "--t": (v - 0.25) / 0.3 } as CSSProperties)}
                        >
                          {v.toFixed(4)}
                        </td>
                      ))}
                      <td className="num">{rowSum(i).toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card
            label="Mean force on each memory"
            meta="Averaged over all 500 steps"
            note={
              <>
                Inhibition is identical for every memory, as the model says it should be. Only the {nameOf(winner).toLowerCase()}{" "}
                receives replay. Its decay is the largest because its activation stays highest, and diffusion moves a little of
                that activation to the other three.
              </>
            }
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Memory</th>
                    {FORCE_KEYS.map((k) => (
                      <th key={k} className="num">
                        {k === "inhibition" ? "Inhibition" : k[0].toUpperCase() + k.slice(1)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {labels.map((l) => (
                    <tr key={l} className={l === winner ? "win" : undefined}>
                      <td>{nameOf(l)}</td>
                      {FORCE_KEYS.map((k) => (
                        <td key={k} className="num">
                          {signed(replay.mean_forces[l][k])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card
            label="Figure 2 · Replay tracking"
            meta="Colab output"
            note={
              <>
                The run's own figure. Its labels are from that run: <code>labrador</code> is the beagle and <code>cute_dog</code> the
                French bulldog. Bottom right, read off the panel: the {nameOf(winner).toLowerCase()}'s pre-softmax score is the
                highest at every replay event, about 640 against about 625 for the next near the start. Multiplied by{" "}
                <N>β</N> = {replay.setup.beta}, a gap that size leaves the others a vanishing share, which is why they print as
                0.0000.
              </>
            }
          >
            <div className="plate">
              <img
                src={figures.softmax}
                alt="Six panels from the replay-tracking run: per-memory activation with only the person staying above zero, the person's replay weight flat at 1 and the others at 0, the force breakdown, energy falling, and pre-softmax scores with the person highest throughout."
              />
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}
