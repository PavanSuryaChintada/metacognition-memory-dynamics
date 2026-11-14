import { recorded, replay } from "../data/results";
import { Card, Section } from "./Layout";

const N = ({ children }: { children: string }) => <span className="nocase">{children}</span>;

type Row = { claim: string; where: string; run: string; result: string; kind: "ok" | "note" | "new" };

/** Paper claim, what the two recorded runs show, and how they line up. Numbers only from the records. */
export function VerdictSection({ num, startEnergy, minDecay }: { num: number; startEnergy: number; minDecay: number }) {
  const f = replay.mean_forces;
  const sweepB = recorded.regimes.sweep_b.runs;
  const drive = (r: { eta: number; replay_prob: number }) => +(r.eta * r.replay_prob).toFixed(4);
  const runaway = sweepB[sweepB.length - 1];
  const bound = runaway.eta / minDecay;
  const diff = Object.values(f).map((m) => m.diffusion);

  const rows: Row[] = [
    {
      claim: "Replay sustains memories against decay",
      where: "§4.2",
      run: `Only the person received replay, a mean force of +${f.person.replay.toFixed(3)} against ${f.person.decay.toFixed(3)} from decay, and it is the only memory left at step ${recorded.steps}.`,
      result: "Matches",
      kind: "ok",
    },
    {
      claim: "Inhibition regulates total activation",
      where: "§2.3",
      run: `The same ${f.person.inhibition.toFixed(5)} on every memory.`,
      result: "Matches",
      kind: "ok",
    },
    {
      claim: "Trajectories stay bounded",
      where: "§3.2",
      run: `The largest ‖a‖, about 6.1 in the runaway regime, stays under the bound of ${bound.toFixed(2)} (from λmin(R) ≥ min α = ${minDecay.toFixed(3)}).`,
      result: "Matches",
      kind: "ok",
    },
    {
      claim: "Collapse, balanced and runaway regimes as replay strengthens",
      where: "§4",
      run: `All three appear, at η·p = ${sweepB.map(drive).join(", ")}, with final ‖a‖ of ${sweepB.map((r) => r.final_norm).join(", ")}.`,
      result: "Matches",
      kind: "ok",
    },
    {
      claim: "The regulatory part only loses energy; replay sits outside it",
      where: "§2.4",
      run: `With replay on, energy falls about 200 times, from ${startEnergy.toFixed(2)} to ${recorded.single_run.final_energy.toFixed(4)}, then levels off.`,
      result: "Consistent",
      kind: "ok",
    },
    {
      claim: "Diffusion spreads activation across similar memories",
      where: "§2.8",
      run: `Mean diffusion forces run from ${Math.min(...diff).toFixed(3)} to +${Math.max(...diff).toFixed(3)}, small next to decay and replay.`,
      result: "Small here",
      kind: "note",
    },
    {
      claim: "Replay selects patterns through a softmax over XᵀXa",
      where: "§2.5",
      run: `At β = ${replay.setup.beta}, one memory took a replay weight of 1.0000 across the replay events measured; the other three got 0.0000.`,
      result: "One winner",
      kind: "new",
    },
  ];

  return (
    <Section
      id="findings"
      label="Findings"
      num={num}
      headline="What the paper predicts, and what this run shows."
      lead="Every entry below comes from the two recorded Colab runs: the notebook and the replay-tracking run."
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="Paper against this run" meta="Ghosh, Jaiswal and Gupta">
            <div className="scroll">
              <table className="verdict">
                <thead>
                  <tr>
                    <th>The paper</th>
                    <th>This run</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.claim}>
                      <td>
                        {r.claim} <span className="muted">{r.where}</span>
                      </td>
                      <td>{r.run}</td>
                      <td>
                        <span className={`tag ${r.kind}`}>{r.result}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <div data-reveal>
            <div className="label" style={{ marginBottom: 14 }}>
              Not tested here
            </div>
            <ul className="points">
              <li>Anything statistical: one seed and four memories, where the paper uses 10 seeds and 200 memories.</li>
              <li>
                Why the person's replay score is the highest. Cosine similarity alone doesn't explain it (the French bulldog's total is
                1.27, the person's 1.22), and the embedding lengths weren't printed.
              </li>
              <li>
                Whether replay spreads across memories with normalised scores or a smaller <N>β</N>.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
}
