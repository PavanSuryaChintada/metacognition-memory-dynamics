import { useEffect, useState } from "react";

export type Route = "home" | "unnormalised" | "normalised";

const parse = (hash: string): Route =>
  hash === "#/unnormalised" ? "unnormalised" : hash === "#/normalised" ? "normalised" : "home";

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
  useEffect(() => {
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
