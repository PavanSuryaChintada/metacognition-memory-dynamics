import type { CSSProperties } from "react";
import { memories } from "../data/results";
import { useIsNarrow, usePinnedProgress } from "../hooks";
import { Tex } from "./Tex";

export const FORCES = [
  {
    key: "decay",
    name: "Decay",
    tex: "-Aa",
    text: "Each memory fades at its own rate α, drawn once from U[0.5, 1.5]. On its own, decay takes every memory to zero.",
  },
  {
    key: "diffusion",
    name: "Diffusion",
    tex: "-\\gamma L a",
    text: "Activation leaks along the similarity graph, from each memory to the ones I-JEPA sees as alike. Edges are cosine similarity, clipped at zero.",
  },
  {
    key: "inhibition",
    name: "Inhibition",
    tex: "-\\delta(\\mathbf{1}^\\top a)\\mathbf{1}",
    text: "The total activation pushes every memory down by the same amount, so the memories compete for one shared budget.",
  },
  {
    key: "replay",
    name: "Replay",
    tex: "\\eta\\, S(t)\\, F(a)",
    text: "On a step where the coin flip lands (probability p), the memories most like the current state are reinforced through a softmax that sums to 1.",
  },
];

/* Isometric plate geometry: a square seen at 2:1, plus a thin side face. */
const W = 150;
const H = 75;
const T = 9;
const rhombus = (cx: number, cy: number, w = W, h = H) => `M${cx},${cy - h}L${cx + w},${cy}L${cx},${cy + h}L${cx - w},${cy}Z`;
const side = (cx: number, cy: number) =>
  `M${cx - W},${cy}L${cx - W},${cy + T}L${cx},${cy + H + T}L${cx + W},${cy + T}L${cx + W},${cy}L${cx},${cy + H}Z`;
// the four memories sit in a diamond on every plate
const SPOTS = [
  [-0.5, 0],
  [0, -0.5],
  [0.5, 0],
  [0, 0.5],
];
const at = (cx: number, cy: number, i: number) => [cx + SPOTS[i][0] * W, cy + SPOTS[i][1] * H] as const;

export function Plates({ active, spread }: { active: number; spread: number }) {
  const cx = 250;
  const gap = 78 + spread * 34;
  const y0 = 100;
  const survivor = memories.findIndex((m) => m.final > 0);

  return (
    <svg className={`plates${active >= 0 ? " has-active" : ""}`} viewBox="0 0 520 560" role="img" aria-label="Four stacked plates, one per force acting on the four memories">
      {[3, 2, 1, 0].map((i) => {
        const cy = y0 + i * gap;
        const on = i === active;
        const dots = SPOTS.map((_, k) => at(cx, cy, k));
        return (
          <g key={i} className={`plate${on ? " on" : ""}`} style={{ "--lift": on ? "-8px" : "0px" } as CSSProperties}>
            <path d={side(cx, cy)} className="plate-side" />
            <path d={rhombus(cx, cy)} className="plate-top" />
            <path d={rhombus(cx, cy, W * 0.82, H * 0.82)} className="plate-inset" />
            {i === 1 &&
              dots.flatMap((a, j) =>
                dots.slice(j + 1).map((b, k) => (
                  <line key={`${j}-${k}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className="plate-edge" />
                )),
              )}
            {i === 2 && <path d={rhombus(cx, cy, W * 0.66, H * 0.66)} className="plate-pool" />}
            {dots.map(([x, y], k) => {
              const m = memories[k];
              const lit = i === 3 && k === survivor;
              return (
                <g key={k}>
                  {i === 0 && <ellipse cx={x} cy={y} rx={22 / m.decay_rate} ry={11 / m.decay_rate} className="plate-halo" />}
                  {lit && <ellipse cx={x} cy={y} rx={16} ry={8} className="plate-pulse" />}
                  <ellipse cx={x} cy={y} rx={6} ry={3.2} className={`plate-dot${lit ? " lit" : ""}`} />
                </g>
              );
            })}
            <text x={cx + W + 14} y={cy + 4} className="plate-label">
              {FORCES[i].name}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Pinned section: plates on the right, the current force's explanation on the left. */
export function ForcesScrolly() {
  const [ref, p] = usePinnedProgress<HTMLDivElement>();
  const narrow = useIsNarrow();
  const active = Math.min(3, Math.floor(p * 4.0001));

  if (narrow) {
    return (
      <div className="forces-static">
        <Plates active={-1} spread={0.4} />
        <ol className="force-list">
          {FORCES.map((f, i) => (
            <li key={f.key} data-reveal>
              <div className="label">
                {String(i + 1).padStart(2, "0")} · {f.name}
              </div>
              <div className="force-tex">
                <Tex>{f.tex}</Tex>
              </div>
              <p>{f.text}</p>
            </li>
          ))}
        </ol>
      </div>
    );
  }

  return (
    <div className="scrolly" ref={ref}>
      <div className="scrolly-sticky">
        <div className="wrap grid scrolly-grid">
          <div className="force-text">
            {FORCES.map((f, i) => (
              <div key={f.key} className={`force-step${i === active ? " on" : ""}`}>
                <div className="label">
                  {String(i + 1).padStart(2, "0")} · {f.name}
                </div>
                <div className="force-tex">
                  <Tex>{f.tex}</Tex>
                </div>
                <p>{f.text}</p>
              </div>
            ))}
            <div className="force-progress" aria-hidden="true">
              {FORCES.map((f, i) => (
                <span key={f.key} className={i <= active ? "done" : undefined} />
              ))}
            </div>
          </div>
          <div className="force-art">
            <Plates active={active} spread={Math.min(1, p * 1.6)} />
          </div>
        </div>
      </div>
    </div>
  );
}
