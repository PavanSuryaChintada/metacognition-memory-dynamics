import { useEffect, useRef, useState } from "react";

export const reducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));

/** Adds `is-in` to every [data-reveal] element the first time it scrolls into view. Re-runs when `page` changes. */
export function useRevealAll(page?: unknown) {
  useEffect(() => {
    const els = [...document.querySelectorAll<HTMLElement>("[data-reveal]")];
    if (reducedMotion()) {
      els.forEach((el) => el.classList.add("is-in"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add("is-in");
            io.unobserve(e.target);
          }
        }
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.08 },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [page]);
}

/** Calls fn on scroll and resize, at most once per frame. */
function useScroll(fn: () => void) {
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => fnRef.current());
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);
}

/** Progress 0..1 through a tall section whose child is position: sticky. */
export function usePinnedProgress<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [p, setP] = useState(0);
  useScroll(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const travel = r.height - window.innerHeight;
    setP(travel > 0 ? clamp(-r.top / travel) : 1);
  });
  return [ref, p] as const;
}

/** Progress 0..1 as an element travels from the bottom of the viewport to `end` (fraction of viewport height). */
export function useEnterProgress<T extends HTMLElement>(end = 0.35) {
  const ref = useRef<T>(null);
  const [p, setP] = useState(0);
  useScroll(() => {
    const el = ref.current;
    if (!el) return;
    const vh = window.innerHeight;
    const top = el.getBoundingClientRect().top;
    setP(clamp((vh - top) / (vh * (1 - end))));
  });
  return [ref, p] as const;
}

/** True while the header line sits over an element marked [data-dark]. */
export function useHeaderOnDark(offset = 32) {
  const [dark, setDark] = useState(false);
  useScroll(() => {
    const zones = document.querySelectorAll<HTMLElement>("[data-dark]");
    let on = false;
    zones.forEach((z) => {
      const r = z.getBoundingClientRect();
      if (r.top <= offset && r.bottom >= offset) on = true;
    });
    setDark(on);
  });
  return dark;
}

/** True once the element has been on screen (for one-time draw-in effects). */
export function useSeen<T extends Element>(threshold = 0.35) {
  const ref = useRef<T>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (reducedMotion()) {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setSeen(true);
          io.disconnect();
        }
      },
      { threshold },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [threshold]);
  return [ref, seen] as const;
}

export function useIsNarrow(px = 860) {
  const [narrow, setNarrow] = useState(() => typeof window !== "undefined" && window.innerWidth <= px);
  useScroll(() => setNarrow(window.innerWidth <= px));
  return narrow;
}
