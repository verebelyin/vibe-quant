import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@": `${import.meta.dirname}/src`,
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
