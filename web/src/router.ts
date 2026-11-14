import { useEffect, useRef, useState } from "react";

export type Route = "home" | "unnormalised" | "normalised" | "normalised-10" | "normalised-beyond" | "normalised-beyond-fix";

const parse = (hash: string): Route =>
  hash === "#/unnormalised"
    ? "unnormalised"
    : hash === "#/normalised"
      ? "normalised"
      : hash === "#/normalised/10"
        ? "normalised-10"
        : hash === "#/normalised/beyond"
          ? "normalised-beyond"
          : hash === "#/normalised/beyond/fix"
            ? "normalised-beyond-fix"
            : "home";

export const isBeyond = (r: Route) => r === "normalised-beyond" || r === "normalised-beyond-fix";
export const isRunB = (r: Route) => r === "normalised" || r === "normalised-10" || isBeyond(r);

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
    // switching between run B's views keeps the reader at the cards they clicked
    if (from && from !== route && isRunB(from) && isRunB(route)) {
      const target = isBeyond(from) && isBeyond(route) ? "beyond-tests" : "experiments";
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
