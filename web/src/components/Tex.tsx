import katex from "katex";
import { useMemo } from "react";

/** KaTeX keeps the maths in the same Computer Modern look as the paper. */
export function Tex({ children, block }: { children: string; block?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: !!block, throwOnError: false, output: "htmlAndMathml" }),
    [children, block],
  );
  return block ? (
    <div className="equation" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span dangerouslySetInnerHTML={{ __html: html }} />
  );
}
