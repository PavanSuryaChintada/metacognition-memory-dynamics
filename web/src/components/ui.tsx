import type { CSSProperties } from "react";

/** Maths inside an uppercase label keeps its case: an uppercased η reads as H. */
export const N = ({ children }: { children: string }) => <span className="nocase">{children}</span>;

/** Stagger index for reveal and entrance animations. */
export const d = (n: number) => ({ "--d": n }) as CSSProperties;
