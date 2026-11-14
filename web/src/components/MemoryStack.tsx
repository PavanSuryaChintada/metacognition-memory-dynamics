import type { CSSProperties } from "react";
import { memories } from "../data/results";
import { useIsNarrow, usePinnedProgress } from "../hooks";

const ease = (t: number) => 1 - Math.pow(1 - t, 3);

function MemoryCard({ m, i, style }: { m: (typeof memories)[number]; i: number; style?: CSSProperties }) {
  return (
    <article className="memory-card" style={style}>
      <div className="memory-card-head">
        <span className="label">({String(i + 1).padStart(2, "0")})</span>
      </div>
      <img src={m.thumb} alt={m.name} width={480} height={360} loading="lazy" />
      <h4>{m.name}</h4>
      <dl>
        <div>
          <dt>Decay rate α</dt>
          <dd>{m.decay_rate.toFixed(2)}</dd>
        </div>
        <div>
          <dt>Starting activation</dt>
          <dd>1</dd>
        </div>
      </dl>
    </article>
  );
}

/** Pinned section: the four memory cards start as a stack and lay out into a row as you scroll. */
export function MemoryStack() {
  const [ref, p] = usePinnedProgress<HTMLDivElement>();
  const narrow = useIsNarrow();

  if (narrow) {
    return (
      <div className="wrap">
        <div className="memory-grid">
          {memories.map((m, i) => (
            <div key={m.label} data-reveal style={{ "--d": i } as CSSProperties}>
              <MemoryCard m={m} i={i} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  const t = ease(Math.min(1, p * 1.35));
  return (
    <div className="stack-scrolly" ref={ref}>
      <div className="stack-sticky">
        <div className="wrap">
          <div className="memory-row">
            {memories.map((m, i) => {
              // stacked: every card pulled to the middle, fanned slightly; laid out: its own column
              const dx = (1.5 - i) * 106 * (1 - t);
              const dy = (i - 1.5) * 10 * (1 - t);
              const rot = (i - 1.5) * 4 * (1 - t);
              return (
                <MemoryCard
                  key={m.label}
                  m={m}
                  i={i}
                  style={{
                    transform: `translate(${dx}%, ${dy}px) rotate(${rot}deg)`,
                    zIndex: 4 - i,
                  }}
                />
              );
            })}
          </div>
          <p className="stack-hint label" style={{ opacity: 1 - t }}>
            Scroll to lay the memories out
          </p>
        </div>
      </div>
    </div>
  );
}
