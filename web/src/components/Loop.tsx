import { reducedMotion, useSeen } from "../hooks";

const R = 128;
const C = 180;
const STEPS = [
  { title: "Higher activation", sub: "aᵢ goes up", angle: -90 },
  { title: "Higher score", sub: "sᵢ = (XᵀXa)ᵢ", angle: 0 },
  { title: "Bigger share", sub: "softmax(βs)ᵢ", angle: 90 },
  { title: "More replay", sub: "+ η·Fᵢ(a)", angle: 180 },
];

/**
 * The rich-get-richer loop that would explain a single winner. Drawn as a hypothesis:
 * the run fits it, but didn't record the replay scores that would test it.
 */
export function Loop() {
  const [ref, seen] = useSeen<HTMLDivElement>(0.3);
  const live = seen && !reducedMotion();
  const circle = `M${C},${C - R}A${R},${R} 0 1 1 ${C - 0.01},${C - R}`;
  return (
    <div ref={ref} className="loop-wrap">
      <svg viewBox="0 0 360 360" className="loop" role="img" aria-label="A loop: higher activation gives a higher replay score, a bigger softmax share, more replay, and higher activation again.">
        <path d={circle} className="loop-ring" />
        {STEPS.map((s) => {
          const rad = (s.angle * Math.PI) / 180;
          const x = C + R * Math.cos(rad);
          const y = C + R * Math.sin(rad);
          return (
            <g key={s.title}>
              <circle cx={x} cy={y} r={5} className="loop-node" />
              <text x={x} y={y + (s.angle === 90 ? 28 : s.angle === -90 ? -30 : -12)} textAnchor={s.angle === 0 ? "end" : s.angle === 180 ? "start" : "middle"} dx={s.angle === 0 ? -14 : s.angle === 180 ? 14 : 0} className="loop-title">
                {s.title}
              </text>
              <text x={x} y={y + (s.angle === 90 ? 46 : s.angle === -90 ? -12 : 6)} textAnchor={s.angle === 0 ? "end" : s.angle === 180 ? "start" : "middle"} dx={s.angle === 0 ? -14 : s.angle === 180 ? 14 : 0} className="loop-sub">
                {s.sub}
              </text>
            </g>
          );
        })}
        {live && (
          <circle r={5} className="loop-pulse">
            <animateMotion path={circle} dur="4.8s" repeatCount="indefinite" />
          </circle>
        )}
      </svg>
    </div>
  );
}
