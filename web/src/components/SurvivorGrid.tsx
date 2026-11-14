import type { CSSProperties } from "react";
import { paperTuned } from "../data/results";

/**
 * Run C's sweep as a grid: rows are γ, columns δ, each cell the number of memories above the
 * survival line at the last step. Tint grows with the count; the paper's cell is dashed.
 */
export function SurvivorGrid({ large = false }: { large?: boolean }) {
  const { gammas, deltas, paper, n, alive_threshold } = paperTuned.setup;
  const cell = (g: number, d: number) => paperTuned.grid.find((c) => c.gamma === g && c.delta === d)!;
  return (
    <div
      className={`sgrid${large ? " sgrid-large" : ""}`}
      role="table"
      aria-label={`Memories above ${alive_threshold} at the last step, for each γ (rows) and δ (columns)`}
    >
      <div className="sg-row sg-head" role="row">
        <span className="sg-corner" role="columnheader">
          γ \ δ
        </span>
        {deltas.map((d) => (
          <span key={d} role="columnheader">
            {d}
          </span>
        ))}
      </div>
      {gammas.map((g) => (
        <div className="sg-row" role="row" key={g}>
          <span className="sg-rowhead" role="rowheader">
            {g}
          </span>
          {deltas.map((d) => {
            const c = cell(g, d);
            const isPaper = g === paper.gamma && d === paper.delta;
            return (
              <span
                key={d}
                role="cell"
                className={`sg-cell${isPaper ? " paper" : ""}${c.alive === 0 ? " none" : ""}`}
                style={{ "--t": c.alive / n } as CSSProperties}
                title={`γ = ${g}, δ = ${d}: ${c.alive} of ${n} above ${alive_threshold}, highest ${c.max}`}
              >
                <b>{c.alive}</b>
                {large && <small>{c.max.toFixed(6)}</small>}
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}
