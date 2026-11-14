import type { ReactNode } from "react";
import { REPO } from "../data/results";

export function Header() {
  return (
    <header className="site-header">
      <div className="wrap">
        <a href="#top" className="wordmark">
          Memory Dynamics <small>on I-JEPA</small>
        </a>
        <nav className="site-nav" aria-label="Sections">
          <a href="#paper">Paper</a>
          <a href="#model">Model</a>
          <a href="#results">Results</a>
          <a href="#theory">Theory</a>
          <a href={REPO} className="keep">
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}

/** A giant chapter word, as a divider between groups of sections. */
export function Chapter({ id, children }: { id?: string; children: ReactNode }) {
  return (
    <div className="wrap chapter-head" id={id}>
      <h2 className="chapter">{children}</h2>
    </div>
  );
}

/** Section row: label in the first column, headline from the second, number on the right. */
export function Section({
  id,
  label,
  num,
  headline,
  lead,
  children,
}: {
  id: string;
  label: string;
  num: number;
  headline: ReactNode;
  lead?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <section className="section" id={id}>
      <div className="wrap grid section-head">
        <span className="label">{label}</span>
        <div className="headline-wrap">
          <h3 className="headline">{headline}</h3>
          {lead && <p className="lead">{lead}</p>}
        </div>
        <span className="num">({String(num).padStart(2, "0")})</span>
      </div>
      {children && <div className="wrap section-body">{children}</div>}
    </section>
  );
}

/** Light card with a label row, used for every figure and table. */
export function Card({
  label,
  meta,
  title,
  children,
  numbers,
  note,
  className = "",
}: {
  label: ReactNode;
  meta?: ReactNode;
  title?: ReactNode;
  children: ReactNode;
  numbers?: ReactNode;
  note?: ReactNode;
  className?: string;
}) {
  return (
    <figure className={`card ${className}`} style={{ margin: 0 }}>
      <div className="card-head">
        <div>
          <div className="label">{label}</div>
          {title && <p className="card-title">{title}</p>}
        </div>
        {meta && <div className="meta">{meta}</div>}
      </div>
      {children}
      {note && <figcaption className="card-note">{note}</figcaption>}
      {numbers && (
        <details className="numbers">
          <summary>Show the numbers</summary>
          <div className="scroll">{numbers}</div>
        </details>
      )}
    </figure>
  );
}
