import { Header } from "./components/Layout";
import { Closing } from "./components/Closing";
import { Home } from "./pages/Home";
import { RunUnnormalised } from "./pages/RunUnnormalised";
import { RunNormalised } from "./pages/RunNormalised";
import { RunPaperScale } from "./pages/RunPaperScale";
import { useRevealAll } from "./hooks";
import { isRunB, isRunC, useRoute } from "./router";

/** Overview with the two run cards, plus one page per run, switched by the URL hash. */
export function App() {
  const route = useRoute();
  useRevealAll(route);
  return (
    <>
      <Header route={route} />
      <main key={isRunB(route) ? "normalised" : isRunC(route) ? "paper" : route}>
        {route === "unnormalised" ? (
          <RunUnnormalised />
        ) : isRunC(route) ? (
          <RunPaperScale view={route === "paper-tuned" ? 2 : 1} />
        ) : isRunB(route) ? (
          <RunNormalised
            view={
              route === "normalised-10" ? 10 : route === "normalised-beyond" ? "beyond" : route === "normalised-beyond-fix" ? "beyond-fix" : 4
            }
          />
        ) : (
          <Home />
        )}
      </main>
      <Closing />
    </>
  );
}
