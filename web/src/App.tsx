import { Header } from "./components/Layout";
import { Closing } from "./components/Closing";
import { Home } from "./pages/Home";
import { RunUnnormalised } from "./pages/RunUnnormalised";
import { RunNormalised } from "./pages/RunNormalised";
import { useRevealAll } from "./hooks";
import { useRoute } from "./router";

/** Overview with the two run cards, plus one page per run, switched by the URL hash. */
export function App() {
  const route = useRoute();
  useRevealAll(route);
  return (
    <>
      <Header route={route} />
      <main key={route === "normalised-10" ? "normalised" : route}>
        {route === "unnormalised" ? (
          <RunUnnormalised />
        ) : route === "normalised" || route === "normalised-10" ? (
          <RunNormalised view={route === "normalised-10" ? 10 : 4} />
        ) : (
          <Home />
        )}
      </main>
      <Closing />
    </>
  );
}
