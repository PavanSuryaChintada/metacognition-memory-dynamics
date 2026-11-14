import type { ReactNode } from "react";
import { Card } from "./Layout";
import { Outcome } from "./Memories";
import { Tex } from "./Tex";
import { d } from "./ui";

/** Top of a run's page: back link, name, headline, the explanation, its replay operator, and the outcome. */
export function RunIntro({
  run,
  name,
  headline,
  explanation,
  formula,
  formulaNote,
  finals,
  outcomeLabel,
  outcomeMeta,
}: {
  run: string;
  name: string;
  headline: string;
  explanation: ReactNode;
  formula: string;
  formulaNote: ReactNode;
  finals?: Record<string, number>;
  outcomeLabel?: string;
  outcomeMeta?: string;
}) {
  return (
    <section className="run-intro" id="top">
      <div className="wrap">
        <a className="back rise" href="#runs" style={d(0)}>
          <span aria-hidden="true">←</span> All runs
        </a>
        <p className="label rise" style={d(1)}>
          {run} · {name}
        </p>
        <h1 className="run-headline rise" style={d(2)}>
          {headline}
        </h1>
        <div className="grid run-explain">
          <div className="run-explain-text rise" style={d(3)}>
            {explanation}
          </div>
          <div className="run-explain-card rise" style={d(4)}>
            <Card label="Replay operator" meta={run}>
              <Tex block>{formula}</Tex>
              <p className="card-note" style={{ marginTop: 4 }}>
                {formulaNote}
              </p>
            </Card>
          </div>
        </div>
        {finals && outcomeLabel && outcomeMeta && (
          <div className="rise" style={d(5)}>
            <Outcome finals={finals} label={outcomeLabel} meta={outcomeMeta} />
          </div>
        )}
      </div>
    </section>
  );
}
