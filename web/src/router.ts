import { useEffect, useRef, useState } from "react";

export type Route = "home" | "unnormalised" | "normalised" | "normalised-10" | "beyond" | "beyond-fix" | "paper" | "paper-tuned";

const parse = (hash: string): Route =>
  hash === "#/unnormalised"
    ? "unnormalised"
    : hash === "#/normalised"
      ? "normalised"
      : hash === "#/normalised/10"
        ? "normalised-10"
        : hash === "#/beyond" || hash === "#/normalised/beyond"
          ? "beyond"
          : hash === "#/beyond/fix" || hash === "#/normalised/beyond/fix"
            ? "beyond-fix"
            : hash === "#/paper"
              ? "paper"
              : hash === "#/paper/tuned"
                ? "paper-tuned"
                : "home";

export const isBeyond = (r: Route) => r === "beyond" || r === "beyond-fix";
export const isRunC = (r: Route) => r === "paper" || r === "paper-tuned";
export const isRunB = (r: Route) => r === "normalised" || r === "normalised-10";

/**
 * Hash routing: "#/unnormalised" and "#/normalised" open a run's page; any other hash is
 * an anchor on the home page, scrolled to once the page has rendered.
 */
export function useRoute() {
  const [hash, setHash] = useState(() => (typeof window === "undefined" ? "" : window.location.hash));
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const route = parse(hash);
  const prev = useRef<Route | null>(null);
  useEffect(() => {
    const from = prev.current;
    prev.current = route;
    // switching between views of the same run keeps the reader at the cards they clicked
    const sameRun =
      from !== null && ((isRunB(from) && isRunB(route)) || (isRunC(from) && isRunC(route)) || (isBeyond(from) && isBeyond(route)));
    if (from && from !== route && sameRun) {
      const target = isRunC(route) ? "paper-tests" : isBeyond(route) ? "beyond-tests" : "experiments";
      const raf = requestAnimationFrame(() => document.getElementById(target)?.scrollIntoView());
      return () => cancelAnimationFrame(raf);
    }
    if (route !== "home" || !hash) {
      window.scrollTo({ top: 0, behavior: "auto" });
      return;
    }
    const id = decodeURIComponent(hash.slice(1));
    const raf = requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView());
    return () => cancelAnimationFrame(raf);
  }, [hash, route]);

  return route;
}
