import { Header } from "./components/Layout";
import { Closing } from "./components/Closing";
import { Home } from "./pages/Home";
import { RunUnnormalised } from "./pages/RunUnnormalised";
import { RunNormalised } from "./pages/RunNormalised";
import { RunPaperScale } from "./pages/RunPaperScale";
import { RunBeyond } from "./pages/RunBeyond";
import { useRevealAll } from "./hooks";
import { isBeyond, isRunB, isRunC, useRoute } from "./router";

/** Overview with the two run cards, plus one page per run, switched by the URL hash. */
export function App() {
  const route = useRoute();
  useRevealAll(route);
  return (
    <>
      <Header route={route} />
      <main key={isRunB(route) ? "normalised" : isRunC(route) ? "paper" : isBeyond(route) ? "beyond" : route}>
        {route === "unnormalised" ? (
          <RunUnnormalised />
        ) : isRunC(route) ? (
          <RunPaperScale view={route === "paper-tuned" ? 2 : 1} />
        ) : isRunB(route) ? (
          <RunNormalised view={route === "normalised-10" ? 10 : 4} />
        ) : isBeyond(route) ? (
          <RunBeyond test={route === "beyond-fix" ? 2 : 1} />
        ) : (
          <Home />
        )}
      </main>
      <Closing />
    </>
  );
}
