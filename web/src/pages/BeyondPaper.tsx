import { figures, fixes, fixMemories, forgetting, forgettingMemories, NOTEBOOKS } from "../data/results";
import { file } from "../data/derived";
import { Card, Section } from "../components/Layout";
import { Tex } from "../components/Tex";
import { N } from "../components/ui";

/** Lower-case a memory name mid-sentence, keeping "French" capitalised. */
const lc = (name: string) => (name.startsWith("French") ? name : name.toLowerCase());

/** Experiment 3 on run B: two cards, one per test, and the chosen test's sections below. */
export function BeyondView({ test }: { test: 1 | 2 }) {
  return (
    <>
      <BeyondTestCards test={test} />
      {test === 1 ? <Test1View /> : <Test2View />}
    </>
  );
}

const fixRet = fixes.retention_pct_printed;

function BeyondTestCards({ test }: { test: 1 | 2 }) {
  return (
    <section className="section" id="beyond-tests">
      <div className="wrap">
        <div className="run-cards beyond-cards">
          <a className={`run-card experiment-card${test === 1 ? " on" : ""}`} href="#/normalised/beyond" aria-current={test === 1 ? "page" : undefined}>
            <div className="run-card-head">
              <span className="label">Test 1</span>
              {test === 1 && <span className="tag new">Showing</span>}
            </div>
            <h2 className="run-title">New memories at full strength</h2>
            <p className="run-desc">
              Five memories at step 0, five more added at step {forgetting.setup.inject_at} at activation 1. The horse photo's link was
              dead, so a random vector stood in for it.
            </p>
            <Timeline />
            <dl className="run-stats">
              <div>
                <dt>Old memories left</dt>
                <dd>
                  {forgetting.survivors.old} of {forgetting.survivors.old_total}{" "}
                  <span className="dd-sub">the one lost is the random vector</span>
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
          <a className={`run-card experiment-card${test === 2 ? " on" : ""}`} href="#/normalised/beyond/fix" aria-current={test === 2 ? "page" : undefined}>
            <div className="run-card-head">
              <span className="label">Test 2</span>
              {test === 2 && <span className="tag new">Showing</span>}
            </div>
            <h2 className="run-title">Adding new memories gently</h2>
            <p className="run-desc">
              The same test with a photo in every slot, run three ways: new memories start at 1, at the mean activation, or warm up over{" "}
              {fixes.setup.warmup_steps} steps.
            </p>
            <RetentionBars />
            <dl className="run-stats">
              <div>
                <dt>Old activation kept</dt>
                <dd>
                  {fixRet.raw}% → {fixRet.mean}% <span className="dd-sub">start at 1 → start at the mean</span>
                </dd>
              </div>
              <div>
                <dt>Old memories left</dt>
                <dd>
                  {fixes.survivors.raw.old} of 5 <span className="dd-sub">in every arm</span>
                </dd>
              </div>
            </dl>
          </a>
        </div>
      </div>
    </section>
  );
}

/** The notebook's retention for each arm, on a 0 to 100% scale. */
function RetentionBars() {
  return (
    <ul className="ret-bars" aria-label="Old activation kept from step 250 to step 500, by arm">
      {fixes.arms.map((a) => (
        <li key={a.id} className={a.id === "raw" ? undefined : "fix"}>
          <span className="ret-name">{a.name}</span>
          <span className="ret-track">
            <span className="ret-fill" style={{ width: `${fixRet[a.id]}%` }} />
          </span>
          <span className="ret-value">{fixRet[a.id]}%</span>
        </li>
      ))}
    </ul>
  );
}

/** A memory's thumbnail, or a marked box when the memory isn't a photo. */
function Thumb({ m }: { m: { thumb?: string } }) {
  return m.thumb ? (
    <img className="thumb" src={m.thumb} alt="" width={48} height={36} />
  ) : (
    <span className="thumb thumb-none" aria-hidden="true">
      no photo
    </span>
  );
}

/* ------------------------------------------------------------------ test 1 */

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
export function Timeline({ large = false }: { large?: boolean }) {
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

function Test1View() {
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
      lead={`Experiments 1 and 2 store all of their memories at step 0. Here five are stored at step 0 and run for ${fg.inject_at} steps. Then five new ones are added at activation ${fg.new_memory_start_activation}, and all ten run on to step ${fg.steps}.`}
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
              <b>With a photo in that slot, nothing dies.</b> Test 2 puts a photo there (the spaniel puppy, still labelled{" "}
              <code>horse</code>) and keeps one arm with new memories starting at 1. In that arm all five old memories stay above{" "}
              {fg.alive_threshold}.
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
              <li>Gentler ways of adding new memories: that is test 2.</li>
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
            <a className="btn btn-text" href="#/normalised/beyond/fix">
              Next: test 2, adding new memories gently <span aria-hidden="true">→</span>
            </a>
          </div>
        </div>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ test 2 */

const fx = fixes.setup;
type ArmId = "raw" | "mean" | "warmup";
const ARMS = fixes.arms;
const fixOld = fixMemories.filter((m) => m.group === "old");
const fixNew = fixMemories.filter((m) => m.group === "new");
const sumOf = (ms: typeof fixMemories, arm: ArmId) => ms.reduce((s, m) => s + m.final[arm], 0);
const S0 = fixes.old_sum_at_250_derived; // old memories' sum just before step 250, the same in every arm
const startMean = S0 / fixOld.length;
const sumAfterFx: Record<ArmId, number> = {
  raw: S0 + fixNew.length * 1,
  mean: S0 + fixNew.length * startMean,
  warmup: S0 + fixNew.length * 0.001,
};
const stepsAfter = fx.steps - fx.inject_at;
/** What decay alone leaves of a memory over steps 250 to 500, with the notebook's Euler step. */
const decayOnly = (alpha: number) => (1 - fx.h * alpha) ** stepsAfter;
const decayRange = [
  decayOnly(Math.max(...fixOld.map((m) => m.decay_rate_printed))),
  decayOnly(Math.min(...fixOld.map((m) => m.decay_rate_printed))),
] as const;
const maxArmGap = Math.max(...fixMemories.map((m) => Math.abs(m.final.mean - m.final.warmup)));
const pairOldFx = fixOld.find((o) => fixNew.some((n) => n.decay_rate_printed === o.decay_rate_printed && o.label === "landscape"))!;
const pairNewFx = fixNew.find((n) => n.decay_rate_printed === pairOldFx.decay_rate_printed)!;
const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;
const times = (a: number, b: number) => `×${(a / b).toFixed(a / b < 1.1 ? 2 : 1)}`;

function Test2View() {
  return (
    <>
      <ArmsSection num={1} />
      <IdeaSection num={2} />
      <FixResultSection num={3} />
      <OldNewSection num={4} />
      <StillGoesSection num={5} />
      <FixFigureSection num={6} />
      <FixFindingsSection num={7} />
      <FixCodeSection num={8} />
    </>
  );
}

function ArmsSection({ num }: { num: number }) {
  return (
    <Section
      id="arms"
      label="The test"
      num={num}
      headline="One test, three ways to add the new memories."
      lead={`Five memories from step 0, five more at step ${fx.inject_at}, the same seed and settings in every arm. Steps 0 to ${fx.inject_at - 1} are identical in all three; only the way the new memories enter changes.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="The three arms"
            meta="5_5forgetting_test_fix.ipynb"
            note={
              <>
                The notebook's header comment also lists a fix that boosts the old memories' replay for a while. The notebook doesn't run
                it.
              </>
            }
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Arm</th>
                    <th>How each new memory starts</th>
                    <th>Printed as</th>
                  </tr>
                </thead>
                <tbody>
                  {ARMS.map((a) => (
                    <tr key={a.id}>
                      <td>
                        <b>{a.name}</b>
                      </td>
                      <td>{a.how}</td>
                      <td className="muted">
                        <code>{a.printed}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card
            label="The memories"
            meta="Every slot is a photo"
            note={
              <>
                This notebook changed two photo links from test 1. The memory it labels <code>horse</code> uses the spaniel puppy photo the
                other runs label <code>puppy</code>, and the one it labels <code>puppy</code> is a new photo of a tabby cat. This page names
                them by what the photos show.
              </>
            }
          >
            <div className="group-strip">
              {[fixOld, fixNew].map((group, gi) => (
                <div key={gi}>
                  <div className="label">{gi === 0 ? "Old, from step 0" : `New, from step ${fx.inject_at}`}</div>
                  <ul>
                    {group.map((m) => (
                      <li key={m.label}>
                        <Thumb m={m} />
                        <span>
                          {m.name}
                          {m.label_in_run !== m.label && (
                            <span className="muted small">
                              {" "}
                              <code>{m.label_in_run}</code>
                            </span>
                          )}
                          <span className="muted small"> α {m.decay_rate_printed.toFixed(2)}</span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <p className="card-note" style={{ marginTop: 14 }}>
              Normalised replay. <N>η</N> = {fx.eta}, <N>β</N> = {fx.beta}, <N>γ</N> = {fx.gamma}, <N>δ</N> = {fx.delta}, h = {fx.h}, p ={" "}
              {fx.replay_prob}, seed {fx.seed}, reset before each arm.
            </p>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function IdeaSection({ num }: { num: number }) {
  return (
    <Section
      id="idea"
      label="The idea"
      num={num}
      headline="New memories shouldn't walk in shouting."
      lead="In plain words: when new memories arrive, they should come in at the same volume as everyone else, not at full volume. In code it is one line: a new memory starts at the current mean activation instead of at 1."
    >
      <div className="grid">
        <div className="indent stack">
          <Card label="The inhibition term" meta="The paper's equation">
            <Tex block>
              {String.raw`\frac{da}{dt} = -Aa - \gamma La \;\underbrace{-\;\delta(\mathbf{1}^\top a)\mathbf{1}}_{\text{inhibition}}\; + \eta S(t)F(a)`}
            </Tex>
            <p className="card-note" style={{ marginTop: 4 }}>
              Inhibition adds up every activation and takes <N>δ</N> times that sum off every memory, the same amount for each. So
              however the new memories enter, the sum they bring with them presses on the old ones too.
            </p>
          </Card>
          <Card
            label="Summed activation at step 250"
            meta="Worked out from the printed values"
            note={`The old memories' sum just before step ${fx.inject_at} isn't printed. It is the same in every arm, and each arm's printed retention and old finals put it at about ${S0.toFixed(3)}, so their mean is about ${startMean.toFixed(3)}. Adding five memories at the mean always doubles the sum.`}
          >
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Arm</th>
                    <th className="num">New memories start at</th>
                    <th className="num">Sum just before</th>
                    <th className="num">Sum just after</th>
                    <th className="num">Inhibition</th>
                  </tr>
                </thead>
                <tbody>
                  {ARMS.map((a) => (
                    <tr key={a.id}>
                      <td>{a.name}</td>
                      <td className="num">{a.id === "raw" ? "1" : a.id === "mean" ? `≈ ${startMean.toFixed(3)}` : "0.001, then ramps up"}</td>
                      <td className="num">≈ {S0.toFixed(3)}</td>
                      <td className="num">≈ {sumAfterFx[a.id].toFixed(3)}</td>
                      <td className={`num big${a.id === "raw" ? "" : " ember-text"}`}>{times(sumAfterFx[a.id], S0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <div className="grid-2">
            <Card label="Start at 1" meta="As in test 1">
              <pre className="code-light">
                <code>{"init_val = 1.0"}</code>
              </pre>
            </Card>
            <Card label="Start at the mean" meta="The one-line change" delay={1}>
              <pre className="code-light">
                <code>{"init_val = self.activations.mean().item()"}</code>
              </pre>
            </Card>
          </div>
          <p className="muted" data-reveal>
            In the figure, starting at 1, the old memories bend down sharply at step {fx.inject_at} and are near 0 within about 60 steps.
            Starting at the mean or warming up, they carry on falling smoothly, with no sharp drop.
          </p>
        </div>
      </div>
    </Section>
  );
}

function FixResultSection({ num }: { num: number }) {
  const oldRaw = sumOf(fixOld, "raw");
  const oldMean = sumOf(fixOld, "mean");
  return (
    <Section
      id="result"
      label="The result"
      num={num}
      headline={`Starting at the mean keeps ${Math.round((oldMean / oldRaw - 1) * 100)}% more of the old activation.`}
      lead={`Retention is the old memories' summed activation at step ${fx.steps} as a share of their sum just before step ${fx.inject_at}, as the notebook computes it. Every memory is above ${fx.alive_threshold} at step ${fx.steps} in every arm.`}
    >
      <div className="grid">
        <div className="indent stack">
          <div className="stats">
            {ARMS.map((a, i) => (
              <Card key={a.id} label={a.name} meta="Old activation kept" delay={i}>
                <div className={`stat-value${a.id === "mean" ? " ember" : ""}`}>{fixRet[a.id]}%</div>
                <div className="stat-sub">Old sum at step {fx.steps}: {sumOf(fixOld, a.id).toFixed(4)}.</div>
              </Card>
            ))}
          </div>
          <Card label={`Activation at step ${fx.steps}`} meta="Each arm">
            <div className="scroll">
              <table className="thumb-table">
                <thead>
                  <tr>
                    <th>Memory</th>
                    <th>Group</th>
                    <th className="num">
                      <N>α</N>
                    </th>
                    {ARMS.map((a) => (
                      <th key={a.id} className="num">
                        {a.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {fixMemories.map((m) => (
                    <tr key={m.label}>
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
                      {ARMS.map((a) => (
                        <td key={a.id} className={`num${a.id === "mean" ? " ember-text" : ""}`}>
                          {m.final[a.id].toFixed(6)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card label="The notebook's summary" meta="As printed">
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th />
                    {ARMS.map((a) => (
                      <th key={a.id} className="num">
                        {a.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ["Old memories left", (id: ArmId) => `${fixes.survivors[id].old} of 5`],
                      ["New memories left", (id: ArmId) => `${fixes.survivors[id].new} of 5`],
                      ["Old activation kept", (id: ArmId) => `${fixRet[id]}%`],
                      ["Highest old memory", (id: ArmId) => fixes.summary_printed.max_old[id].toFixed(4)],
                      ["Highest new memory", (id: ArmId) => fixes.summary_printed.max_new[id].toFixed(4)],
                      ["Old mean ÷ new mean", (id: ArmId) => fixes.summary_printed.old_new_ratio[id].toFixed(2)],
                    ] as const
                  ).map(([name, value]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      {ARMS.map((a) => (
                        <td key={a.id} className="num">
                          {value(a.id)}
                        </td>
                      ))}
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

function OldNewSection({ num }: { num: number }) {
  const r = fixes.summary_printed.old_new_ratio;
  return (
    <Section
      id="old-new"
      label="Old against new"
      num={num}
      headline="Old and new end at about the same level."
      lead={`Starting at 1, the old memories end at about a third of the new ones on average (old mean ÷ new mean ${r.raw.toFixed(2)}). Starting at the mean it is ${r.mean.toFixed(2)}; warming up, ${r.warmup.toFixed(2)}.`}
    >
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Same decay rate, one old and one new"
            meta={
              <>
                <N>α</N> = {pairOldFx.decay_rate_printed.toFixed(2)}
              </>
            }
          >
            <div className="scroll">
              <table className="thumb-table">
                <thead>
                  <tr>
                    <th>Arm</th>
                    <th className="num">
                      <span className="thumb-name">
                        <Thumb m={pairOldFx} />
                        {pairOldFx.name}, old
                      </span>
                    </th>
                    <th className="num">
                      <span className="thumb-name">
                        <Thumb m={pairNewFx} />
                        {pairNewFx.name}, new
                      </span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {ARMS.map((a) => (
                    <tr key={a.id}>
                      <td>{a.name}</td>
                      <td className="num big">{pairOldFx.final[a.id].toFixed(4)}</td>
                      <td className="num big">{pairNewFx.final[a.id].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <p className="muted" data-reveal>
            Starting at 1, the newer memory ends {(pairNewFx.final.raw / pairOldFx.final.raw).toFixed(1)} times higher. Starting at the mean,
            the old one ends higher. Warmup ends within {maxArmGap.toFixed(4)} of starting at the mean for every memory.
          </p>
        </div>
      </div>
    </Section>
  );
}

function StillGoesSection({ num }: { num: number }) {
  return (
    <Section
      id="still-goes"
      label="What's left"
      num={num}
      headline="Most of the old activation still goes."
      lead={`Even in the best arm the old memories keep ${fixRet.mean}% of their summed activation. Decay on its own would take a lot of that, and this run has no arm without new memories to compare against.`}
    >
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>Decay alone.</b> Over steps {fx.inject_at} to {fx.steps}, the decay term on its own multiplies a memory by (1 − h
              <N>α</N>)<sup>{stepsAfter}</sup>: about {decayRange[1].toFixed(2)} for <N>α</N> ={" "}
              {Math.min(...fixOld.map((m) => m.decay_rate_printed)).toFixed(2)} down to {decayRange[0].toFixed(2)} for <N>α</N> ={" "}
              {Math.max(...fixOld.map((m) => m.decay_rate_printed)).toFixed(2)}. So with decay alone, the old memories' sum would keep
              between {pct1(decayRange[0])} and {pct1(decayRange[1])}.
            </li>
            <li>
              <b>Against the arms.</b> Starting at the mean keeps {fixRet.mean}%, above anything decay alone would leave. Starting at 1
              keeps {fixRet.raw}%, inside that range. So the loss that remains in the gentler arms is no bigger than ordinary decay.
            </li>
            <li>
              <b>Not tested.</b> A run with no new memories, which would show how much of the drop the new memories cause. Scaling{" "}
              <N>δ</N> with the number of memories (<N>δ</N>/N), an idea for keeping inhibition per memory the same as N grows. Boosting
              the old memories' replay after new ones arrive. Other seeds.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

function FixFigureSection({ num }: { num: number }) {
  return (
    <Section id="figure" label="Figure" num={num} headline="All three arms, step by step.">
      <div className="grid">
        <div className="indent stack">
          <Card
            label="Figure · Three ways to add the new memories"
            meta="Colab output"
            note={
              <>
                The notebook's own figure. Columns: start at 1, start at the mean, warm up. Top row: activations; solid lines are old
                memories, dashed lines new, and the red line is step {fx.inject_at}. Bottom row: each old memory just before step{" "}
                {fx.inject_at} and at step {fx.steps}, with the notebook's retention rounded to whole percent. Its labels are from that run:{" "}
                <code>labrador</code> is the beagle, <code>bulldog</code> the French bulldog, <code>coffee</code> the armchair,{" "}
                <code>horse</code> the spaniel puppy and <code>puppy</code> the tabby cat.
              </>
            }
          >
            <div className="plate">
              <img
                src={figures.fixes}
                alt="Six panels in three columns. Starting at 1: new memories enter at 1 and the old ones drop to near zero soon after step 250. Starting at the mean and warming up: new memories enter near the old ones' level and everything settles together near 0.01 to 0.02. Bottom row: old memories before step 250 and at step 500, with retention 15%, 21% and 21%."
              />
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}

function FixFindingsSection({ num }: { num: number }) {
  return (
    <Section id="findings" label="Findings" num={num} headline="What it shows, and what it doesn't.">
      <div className="grid">
        <div className="indent two-col">
          <div data-reveal>
            <div className="label">Shows</div>
            <ul className="points">
              <li>
                With a photo in every slot, no memory drops below {fx.alive_threshold} in any arm, starting at 1 included. Test 1's one
                loss was the random vector.
              </li>
              <li>
                Starting new memories at the mean keeps {fixRet.mean}% of the old activation against {fixRet.raw}% at 1, and old and new
                end at about the same level.
              </li>
              <li>Warming up over {fx.warmup_steps} steps gives almost the same numbers as starting at the mean, which is the simpler change.</li>
            </ul>
          </div>
          <div data-reveal>
            <div className="label">Doesn't show</div>
            <ul className="points">
              <li>How much of the old memories' drop the new memories cause: no arm leaves them out.</li>
              <li>That the remaining drop is a capacity limit. It is within what decay alone would do.</li>
              <li>Whether <N>δ</N>/N scaling or extra replay for old memories helps: neither was run.</li>
              <li>Anything statistical: one seed.</li>
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
}

function FixCodeSection({ num }: { num: number }) {
  return (
    <Section id="run-code" label="The code" num={num} headline="Where these numbers come from.">
      <div className="grid">
        <div className="indent stack">
          <ul className="points" data-reveal>
            <li>
              <b>The executed notebook.</b>{" "}
              <a href={NOTEBOOKS.forgettingFixes}>notebooks/colab/normalised/beyond_paper/5_5forgetting_test_fix.ipynb</a>, with its code,
              printed output and figure as they ran in Colab.
            </li>
            <li>
              <b>Recorded copies.</b> Printed output in{" "}
              <a href={file("results/forgetting_fixes_output.txt")}>results/forgetting_fixes_output.txt</a>, numbers in{" "}
              <a href={file("results/forgetting_fixes.json")}>results/forgetting_fixes.json</a>.
            </li>
          </ul>
          <div className="btn-row">
            <a className="btn btn-primary" href={NOTEBOOKS.forgettingFixes}>
              Open the Colab notebook on GitHub
            </a>
            <a className="btn btn-text" href="#/normalised/beyond">
              <span aria-hidden="true">←</span> Test 1
            </a>
          </div>
        </div>
      </div>
    </Section>
  );
}
