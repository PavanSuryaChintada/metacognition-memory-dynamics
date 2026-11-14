import { useEffect, useState } from "react";
import { memories, recorded } from "../data/results";
import { Card } from "./Layout";

function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Figure 1: each memory's activation at the start (1) and after the run. On load each
 * bar runs once from its start value to its recorded end value; the in-between is a
 * transition, not data. Memories that end at exactly 0 turn grey.
 */
export function Outcome() {
  const [done, setDone] = useState(prefersReducedMotion);
  useEffect(() => {
    if (done) return;
    let id = requestAnimationFrame(() => (id = requestAnimationFrame(() => setDone(true))));
    return () => cancelAnimationFrame(id);
  }, [done]);

  return (
    <Card
      className="outcome-card"
      label={`Figure 1 · Activation after ${recorded.steps} steps`}
      meta={`Colab run · seed ${recorded.seed} · η = ${recorded.single_run.eta}, p = ${recorded.single_run.replay_prob}`}
    >
      <ol className="outcome" aria-label="Activation of each memory at the end of the run">
        {memories.map((m) => {
          const survived = m.final > 0;
          const state = done ? (survived ? "survived" : "forgotten") : survived ? "survived" : undefined;
          return (
            <li key={m.label} className={state}>
              <img src={m.thumb} alt={m.name} width={480} height={360} />
              <div className="outcome-name">{m.name}</div>
              <div
                className="bar"
                role="meter"
                aria-valuemin={0}
                aria-valuemax={1}
                aria-valuenow={m.final}
                aria-label={`${m.name}: activation ${m.final} at the end`}
              >
                <div className="bar-fill" style={{ width: `${(done ? m.final : 1) * 100}%` }} />
              </div>
              <div className="outcome-value">
                <strong>{survived ? m.final.toFixed(4) : "0"}</strong>
                <span>α = {m.decay_rate.toFixed(2)}</span>
              </div>
            </li>
          );
        })}
      </ol>
    </Card>
  );
}
