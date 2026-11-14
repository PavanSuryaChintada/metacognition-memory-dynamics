import { forgetting } from "../data/results";
import { RunIntro } from "../components/RunIntro";
import { BeyondView } from "./BeyondPaper";

/** Tests beyond the paper: new memories added partway through a run, with run B's normalised replay. */
export function RunBeyond({ test }: { test: 1 | 2 }) {
  const fg = forgetting.setup;
  return (
    <>
      <RunIntro
        run="Beyond the paper"
        name="New memories mid-run"
        headline="What happens when new memories arrive partway through."
        explanation={
          <>
            <p>
              The paper's experiments, and the other runs on this site, store every memory at step 0. These two tests store five at step
              0, run for {fg.inject_at} steps, then add five more and run on to step {fg.steps}. Both use run B's normalised replay.
            </p>
            <p>
              Test 1 adds the new memories at full strength, activation 1. Test 2 runs the same thing three ways: new memories start at 1,
              at the mean activation, or warm up over 50 steps. Pick one.
            </p>
          </>
        }
        formula={String.raw`F(a) = \mathrm{softmax}\big(\beta\, \hat X^\top \hat X a\big)`}
        formulaNote="Run B's normalised replay, used in both tests. What changes is when the memories arrive."
      />
      <BeyondView test={test} />
    </>
  );
}
