import type { CSSProperties, ReactNode } from "react";
import { REPO } from "../data/results";
import { useHeaderOnDark } from "../hooks";
import { isBayes, isBeyond, isRunB, isRunC, type Route } from "../router";

/** The project mark: three memories fading, one held by replay. */
export function Mark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="mark">
      <path d="M4 7c5 0 7 17 24 17" fill="none" stroke="currentColor" strokeOpacity=".35" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M4 7c5 0 6 13 24 13" fill="none" stroke="currentColor" strokeOpacity=".55" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M4 7c4 0 5 7 24 7" fill="none" stroke="var(--ember)" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="28" cy="14" r="2.6" fill="var(--ember)" />
    </svg>
  );
}

export function Header({ route }: { route: Route }) {
  const dark = useHeaderOnDark();
  const onRun = route !== "home";
  return (
    <header className={`site-header${dark ? " on-dark" : ""}`}>
      <div className="wrap header-row">
        <a href="#top" className="wordmark">
          Memory Dynamics
        </a>
        <a href={onRun ? "#runs" : "#top"} className="header-mark" aria-label="Overview">
          <Mark />
        </a>
        <nav className="site-nav" aria-label="Sections">
          {onRun ? (
            <>
              <a href="#runs" className="keep">
                All runs
              </a>
              <a href="#/unnormalised" aria-current={route === "unnormalised" ? "page" : undefined}>
                Run A
              </a>
              <a href="#/normalised" aria-current={isRunB(route) ? "page" : undefined}>
                Run B
              </a>
              <a href="#/paper" aria-current={isRunC(route) ? "page" : undefined}>
                Run C
              </a>
              <a href="#/beyond" aria-current={isBeyond(route) ? "page" : undefined}>
                Beyond
              </a>
              <a href="#/bayes" aria-current={isBayes(route) ? "page" : undefined}>
                Confidence
              </a>
            </>
          ) : (
            <>
              <a href="#runs">Runs</a>
              <a href="#model">Model</a>
              <a href="#how-it-runs">How it runs</a>
              <a href="#paper">Paper</a>
            </>
          )}
          <a href={REPO} className="keep">
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}

/** A giant chapter word, as a divider between groups of sections. */
export function Chapter({ id, children, sub }: { id?: string; children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="wrap chapter-head" id={id}>
      <h2 className="chapter" data-reveal>
        {children}
      </h2>
      {sub && (
        <div className="chapter-sub" data-reveal style={{ "--d": 1 } as CSSProperties}>
          {sub}
        </div>
      )}
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
      <div className="wrap grid section-head" data-reveal>
        <span className="label">{label}</span>
        <div className="headline-wrap">
          <h3 className="headline">{headline}</h3>
          {lead && <div className="lead">{lead}</div>}
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
  delay = 0,
}: {
  label: ReactNode;
  meta?: ReactNode;
  title?: ReactNode;
  children: ReactNode;
  numbers?: ReactNode;
  note?: ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <figure className={`card ${className}`} style={{ margin: 0, "--d": delay } as CSSProperties} data-reveal>
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
