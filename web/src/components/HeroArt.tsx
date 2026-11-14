import { memories, recorded } from "../data/results";
import { reducedMotion } from "../hooks";

/**
 * Background construction lines: column guides across the hero, and in the right margin
 * one hairline per memory showing decay alone, a(t) = exp(-alpha t), over the run's
 * 500 steps (t from 0 to 5). These use only the inputs (the four decay rates), so the
 * overview stays free of either run's results.
 */
export function HeroArt() {
  const W = 1440;
  const H = 640;
  const cols = 6;
  const gx0 = 96;
  const gx1 = W - 96;
  // the small diagram, beside the subtitle and clear of the headline
  const x0 = 1090;
  const x1 = 1390;
  const top = 392;
  const base = 560;
  const span = base - top;
  const tEnd = recorded.steps * recorded.base_params.h;

  const curves = memories.map((m) => {
    const pts: string[] = [];
    for (let s = 0; s <= 80; s++) {
      const u = s / 80;
      const a = Math.exp(-m.decay_rate * u * tEnd);
      pts.push(`${(x0 + u * (x1 - x0)).toFixed(1)},${(base - a * span).toFixed(1)}`);
    }
    return { key: m.label, d: `M${pts.join("L")}`, rate: m.decay_rate };
  });
  const slowest = curves.reduce((a, b) => (b.rate < a.rate ? b : a));

  return (
    <svg className="hero-art" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMin slice" aria-hidden="true">
      {Array.from({ length: cols + 1 }, (_, c) => {
        const x = gx0 + (c * (gx1 - gx0)) / cols;
        return <line key={c} className="guide" x1={x} x2={x} y1={0} y2={H} />;
      })}
      <line className="guide" x1={0} x2={W} y1={150} y2={150} />
      <g className="diagram">
        <line className="guide strong" x1={x0} x2={x0} y1={top - 16} y2={base} />
        <line className="guide strong" x1={x0} x2={x1} y1={base} y2={base} />
        <line className="guide" x1={x0 - 6} x2={x0} y1={top} y2={top} />
        {curves.map((c) => (
          <path key={c.key} d={c.d} pathLength={1} className="curve" />
        ))}
        {!reducedMotion() && (
          <circle className="traveller" r={2.6}>
            <animateMotion path={slowest.d} dur="7s" begin="2.6s" repeatCount="indefinite" />
          </circle>
        )}
      </g>
    </svg>
  );
}
