import { memories } from "../data/results";

/**
 * Background construction lines: column guides across the hero, and in the right
 * margin a small diagram with one hairline per memory, starting at activation 1 and
 * settling at that memory's recorded final value. Start and end are the run's real
 * numbers; the shape in between is schematic, so this is decoration, not a chart.
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

  const curves = memories.map((m) => {
    const k = 3.2 + m.decay_rate * 2.4;
    const pts: string[] = [];
    for (let s = 0; s <= 80; s++) {
      const u = s / 80;
      const a = m.final + (1 - m.final) * Math.exp(-k * u);
      pts.push(`${(x0 + u * (x1 - x0)).toFixed(1)},${(base - a * span).toFixed(1)}`);
    }
    return { key: m.label, d: `M${pts.join("L")}`, survivor: m.final > 0, end: base - m.final * span };
  });

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
          <path key={c.key} d={c.d} pathLength={1} className={`curve${c.survivor ? " survivor" : ""}`} />
        ))}
        {curves
          .filter((c) => c.survivor)
          .map((c) => (
            <circle key={c.key} className="dot" cx={x1} cy={c.end} r={3.5} />
          ))}
      </g>
    </svg>
  );
}
