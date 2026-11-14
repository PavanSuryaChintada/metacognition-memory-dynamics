import { useEffect, useRef, useState } from "react";
import { useSeen } from "../hooks";

type Point = { name: string; drive: number; norm: number };

const TICKS = [0.001, 0.01, 0.1, 1, 10];
const fmt = (v: number) => String(v);

/**
 * Final ‖a‖ against average replay drive η·p, both on log scales. The printed sweep is
 * the line with labelled points; the earlier sweep's values were read off its figure,
 * so they are drawn as unlabelled rings.
 */
export function DriveChart({ exact, approx, start }: { exact: Point[]; approx: Point[]; start: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [seenRef, seen] = useSeen<SVGSVGElement>(0.4);
  const [width, setWidth] = useState(880);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.round(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const height = width < 520 ? 300 : 360;
  const m = { l: 52, r: width < 520 ? 16 : 120, t: 20, b: 52 };
  const W = width - m.l - m.r;
  const H = height - m.t - m.b;
  const lx = (v: number) => m.l + ((Math.log10(v) + 3) / 4.3) * W; // 0.001 .. 20
  const ly = (v: number) => m.t + H - ((Math.log10(v) + 3) / 4.3) * H;
  const path = exact.map((p, i) => `${i ? "L" : "M"}${lx(p.drive).toFixed(1)},${ly(p.norm).toFixed(1)}`).join("");

  return (
    <div className="chart" ref={ref}>
      <div className="legend" style={{ marginBottom: 10 }}>
        <span>
          <i className="key-line" /> Final settings, printed values
        </span>
        {approx.length > 0 && (
          <span>
            <i className="key-ring" /> First sweep, read off its figure
          </span>
        )}
      </div>
      <svg ref={seenRef} className={seen ? "is-drawn" : undefined} width={width} height={height} role="img" aria-label="Final activation size against replay drive, log scales. It rises from 0.0048 at drive 0.0025 to 0.5003 at 0.8 and 5.7369 at 9.">
        {TICKS.map((t) => (
          <g key={`y${t}`}>
            <line className="gridline" x1={m.l} x2={m.l + W} y1={ly(t)} y2={ly(t)} />
            <text className="tick" x={m.l - 8} y={ly(t)} dy="0.32em" textAnchor="end">
              {fmt(t)}
            </text>
          </g>
        ))}
        {TICKS.map((t) => (
          <text key={`x${t}`} className="tick" x={lx(t)} y={m.t + H + 20} textAnchor="middle">
            {fmt(t)}
          </text>
        ))}
        <line className="axis" x1={m.l} x2={m.l + W} y1={m.t + H} y2={m.t + H} />
        <text className="axis-title" x={m.l + W} y={height - 6} textAnchor="end">
          Replay drive η·p
        </text>
        <text className="axis-title" x={m.l} y={m.t - 6}>
          ‖a‖ at step 500
        </text>

        <line x1={m.l} x2={m.l + W} y1={ly(start)} y2={ly(start)} stroke="var(--faint)" strokeWidth={1} />
        <text className="tick" x={m.l + 6} y={ly(start) - 6}>
          start, ‖a(0)‖ = {start}
        </text>

        {approx.map((p) => (
          <circle key={`a${p.name}`} cx={lx(p.drive)} cy={ly(p.norm)} r={4.5} fill="var(--bg)" stroke="var(--faint)" strokeWidth={1.5}>
            <title>{`${p.name}, first sweep: η·p = ${p.drive}, ‖a‖ ≈ ${p.norm} (read off the figure)`}</title>
          </circle>
        ))}
        <path d={path} pathLength={1} className="drive-line" fill="none" stroke="var(--ember)" strokeWidth={2} strokeLinejoin="round" />
        {exact.map((p) => {
          const x = lx(p.drive);
          const flip = x > width - m.r - 70; // near the right edge: label on the left
          const low = ly(p.norm) + 38 > m.t + H; // near the axis: label above
          const tx = flip ? x - 10 : x + 10;
          const anchor = flip ? "end" : "start";
          const above = flip || low;
          return (
            <g key={p.name} className="drive-point" style={{ transitionDelay: `${0.5 + exact.indexOf(p) * 0.45}s` }}>
              <circle cx={x} cy={ly(p.norm)} r={5} fill="var(--ember)" stroke="var(--bg)" strokeWidth={2}>
                <title>{`${p.name}: η·p = ${p.drive}, ‖a‖ = ${p.norm}`}</title>
              </circle>
              {/* below-right normally; above near the axis or the right edge, clear of the rising line */}
              <text x={tx} y={ly(p.norm) + (above ? -24 : 20)} fontSize={13} fill="var(--ink)" fontWeight={560} textAnchor={anchor}>
                {p.norm}
              </text>
              <text x={tx} y={ly(p.norm) + (above ? -10 : 34)} className="tick" textAnchor={anchor}>
                {p.name}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
