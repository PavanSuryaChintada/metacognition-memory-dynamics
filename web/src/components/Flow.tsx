import { reducedMotion, useSeen } from "../hooks";

type Node = { x: number; y: number; w: number; h: number; title: string; sub: string; accent?: boolean };

function Box({ n }: { n: Node }) {
  return (
    <g>
      <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={10} className={`flow-node${n.accent ? " accent" : ""}`} />
      <text x={n.x + 14} y={n.y + n.h / 2 - 4} className="flow-title">
        {n.title}
      </text>
      <text x={n.x + 14} y={n.y + n.h / 2 + 14} className="flow-sub">
        {n.sub}
      </text>
    </g>
  );
}

function Pulse({ path, dur, begin = "0s" }: { path: string; dur: string; begin?: string }) {
  return (
    <circle r={4} className="flow-pulse">
      <animateMotion path={path} dur={dur} begin={begin} repeatCount="indefinite" rotate="auto" />
    </circle>
  );
}

const LOOP_ITEMS = ["decay", "diffusion", "inhibition", "replay (coin flip)"];

function Horizontal({ live }: { live: boolean }) {
  const y = 112;
  const h = 70;
  const nodes: Node[] = [
    { x: 0, y, w: 132, h, title: "Photo", sub: "Unsplash, RGB" },
    { x: 172, y, w: 150, h, title: "I-JEPA", sub: "ViT-H/14, frozen" },
    { x: 362, y, w: 150, h, title: "Embedding", sub: "1,280 numbers" },
    { x: 552, y, w: 150, h, title: "Memory store", sub: "a = 1, own α" },
  ];
  const loop: Node = { x: 742, y: y - 22, w: 160, h: h + 44, title: "Step", sub: "", accent: true };
  const outs = [
    { y: 30, title: "Retrieval", sub: "cos × activation" },
    { y: 128, title: "Figures", sub: "matplotlib" },
    { y: 226, title: "metrics.json", sub: "every number" },
  ];
  const mid = y + h / 2;
  const main = `M132,${mid}L742,${mid}`;
  return (
    <svg className="flow flow-h" viewBox="0 0 1110 310" role="img" aria-label="Flow: photo, I-JEPA, embedding, memory store, a step loop run 500 times, then retrieval, figures and metrics.">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,1L9,5L0,9" fill="none" className="flow-arrowhead" />
        </marker>
      </defs>
      {nodes.slice(0, -1).map((n, i) => (
        <line key={i} x1={n.x + n.w} y1={mid} x2={nodes[i + 1].x - 4} y2={mid} className="flow-link" markerEnd="url(#arrow)" />
      ))}
      <line x1={702} y1={mid} x2={738} y2={mid} className="flow-link" markerEnd="url(#arrow)" />
      {nodes.map((n) => (
        <Box key={n.title} n={n} />
      ))}
      <rect x={loop.x} y={loop.y} width={loop.w} height={loop.h} rx={10} className="flow-node accent" />
      <text x={loop.x + 14} y={loop.y + 24} className="flow-title">
        Step × 500
      </text>
      {LOOP_ITEMS.map((t, i) => (
        <text key={t} x={loop.x + 14} y={loop.y + 48 + i * 17} className="flow-sub">
          {i === 3 ? "+ " : "− "}
          {t}
        </text>
      ))}
      <path
        d={`M${loop.x + 40},${loop.y}C${loop.x + 40},${loop.y - 44} ${loop.x + 120},${loop.y - 44} ${loop.x + 120},${loop.y - 4}`}
        className="flow-link"
        markerEnd="url(#arrow)"
      />
      <text x={loop.x + 80} y={loop.y - 46} textAnchor="middle" className="flow-sub">
        h = 0.01, seed 42
      </text>
      {outs.map((o) => {
        const ox = 962;
        const oy = o.y + 22;
        return (
          <g key={o.title}>
            <path d={`M${loop.x + loop.w},${mid}C${ox - 30},${mid} ${ox - 40},${oy} ${ox - 4},${oy}`} className="flow-link" markerEnd="url(#arrow)" />
            <Box n={{ x: ox, y: o.y, w: 148, h: 44, title: o.title, sub: o.sub }} />
          </g>
        );
      })}
      {live && (
        <>
          <Pulse path={main} dur="3.2s" />
          <Pulse path={`M${loop.x + loop.w},${mid}C932,${mid} 922,152 958,152`} dur="3.2s" begin="1.6s" />
        </>
      )}
    </svg>
  );
}

function Vertical({ live }: { live: boolean }) {
  const x = 40;
  const w = 280;
  const h = 58;
  const steps = [
    { title: "Photo", sub: "Unsplash, RGB" },
    { title: "I-JEPA", sub: "ViT-H/14, frozen" },
    { title: "Embedding", sub: "1,280 numbers" },
    { title: "Memory store", sub: "a = 1, own α" },
  ];
  const ys = steps.map((_, i) => i * 92);
  const loopY = 4 * 92;
  const outY = loopY + 160;
  return (
    <svg className="flow flow-v" viewBox="0 0 360 640" role="img" aria-label="Flow: photo, I-JEPA, embedding, memory store, a step loop run 500 times, then retrieval, figures and metrics.">
      <defs>
        <marker id="arrow-v" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,1L9,5L0,9" fill="none" className="flow-arrowhead" />
        </marker>
      </defs>
      {ys.map((y, i) => (
        <line key={i} x1={180} y1={y + h} x2={180} y2={(i < 3 ? ys[i + 1] : loopY) - 4} className="flow-link" markerEnd="url(#arrow-v)" />
      ))}
      {steps.map((s, i) => (
        <Box key={s.title} n={{ x, y: ys[i], w, h, title: s.title, sub: s.sub }} />
      ))}
      <rect x={x} y={loopY} width={w} height={112} rx={10} className="flow-node accent" />
      <text x={x + 14} y={loopY + 26} className="flow-title">
        Step × 500
      </text>
      {LOOP_ITEMS.map((t, i) => (
        <text key={t} x={x + 14} y={loopY + 50 + i * 17} className="flow-sub">
          {i === 3 ? "+ " : "− "}
          {t}
        </text>
      ))}
      {["Retrieval", "Figures", "metrics.json"].map((t, i) => {
        const bx = 8 + i * 118;
        return (
          <g key={t}>
            <path d={`M180,${loopY + 112}C180,${outY - 20} ${bx + 52},${outY - 30} ${bx + 52},${outY - 4}`} className="flow-link" markerEnd="url(#arrow-v)" />
            <rect x={bx} y={outY} width={108} height={44} rx={10} className="flow-node" />
            <text x={bx + 54} y={outY + 27} textAnchor="middle" className="flow-title small">
              {t}
            </text>
          </g>
        );
      })}
      {live && <Pulse path={`M180,${h}L180,${loopY - 4}`} dur="3s" />}
    </svg>
  );
}

export function Flow() {
  const [ref, seen] = useSeen<HTMLDivElement>(0.25);
  const live = seen && !reducedMotion();
  return (
    <div ref={ref} className={`flow-wrap${seen ? " is-drawn" : ""}`}>
      <Horizontal live={live} />
      <Vertical live={live} />
    </div>
  );
}
