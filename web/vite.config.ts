import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The site reads ../results (metrics.json and the Colab figures) directly, so the
// numbers on the page are the recorded ones with no copy in between.
export default defineConfig({
  plugins: [react()],
  server: { fs: { allow: [".."] } },
});
