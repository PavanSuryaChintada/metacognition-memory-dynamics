import { useEffect, useState } from "react";
import { memories } from "../data/results";
import { Card } from "./Layout";

function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Each memory's activation at the start (1) and at the end of a run. On load each bar runs
 * once from its start value to its recorded end value; the in-between is a transition, not
 * data. Memories that end at exactly 0 turn grey.
 */
export function Outcome({ finals, label, meta }: { finals: Record<string, number>; label: string; meta: string }) {
  const [done, setDone] = useState(prefersReducedMotion);
  useEffect(() => {
    if (done) return;
    let id = requestAnimationFrame(() => (id = requestAnimationFrame(() => setDone(true))));
    return () => cancelAnimationFrame(id);
  }, [done]);

  return (
    <Card className="outcome-card" label={label} meta={meta}>
      <ol className="outcome" aria-label={label}>
        {memories.map((m) => {
          const v = finals[m.label] ?? 0;
          const alive = v > 0;
          const state = done ? (alive ? "survived" : "forgotten") : alive ? "survived" : undefined;
          return (
            <li key={m.label} className={state}>
              <img src={m.thumb} alt={m.name} width={480} height={360} />
              <div className="outcome-name">{m.name}</div>
              <div
                className="bar"
                role="meter"
                aria-valuemin={0}
                aria-valuemax={1}
                aria-valuenow={v}
                aria-label={`${m.name}: activation ${v} at the end`}
              >
                <div className="bar-fill" style={{ width: `${(done ? v : 1) * 100}%` }} />
              </div>
              <div className="outcome-value">
                <strong>{alive ? v.toFixed(4) : "0"}</strong>
                <span>α = {m.decay_rate.toFixed(2)}</span>
              </div>
            </li>
          );
        })}
      </ol>
    </Card>
  );
}

/** Compact version for the run cards: thumbnails with bars on a shared scale. */
export function MiniBars({ finals, scale }: { finals: Record<string, number>; scale: number }) {
  return (
    <ol className="mini-bars" aria-label={`Activation at step 500, bars scaled 0 to ${scale}`}>
      {memories.map((m) => {
        const v = finals[m.label] ?? 0;
        return (
          <li key={m.label} className={v > 0 ? "alive" : "gone"}>
            <img src={m.thumb} alt="" width={96} height={72} />
            <span className="mini-name">{m.name}</span>
            <span className="mini-track">
              <span className="mini-fill" style={{ width: `${Math.min(1, v / scale) * 100}%` }} />
            </span>
            <span className="mini-value">{v > 0 ? v.toFixed(3) : "0"}</span>
          </li>
        );
      })}
    </ol>
  );
}
