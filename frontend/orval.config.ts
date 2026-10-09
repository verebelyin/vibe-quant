// orval pinned to 8.4.0 (exact): 8.39 generates GET endpoints as mutations with this config and breaks callers — upgrade deliberately (bead vibe-quant-4sagi).
import { defineConfig } from "orval";

export default defineConfig({
  vibeQuant: {
    input: {
      target: "./openapi.json",
    },
    output: {
      mode: "tags-split",
      target: "src/api/generated",
      schemas: "src/api/generated/models",
      client: "react-query",
      override: {
        mutator: {
          path: "src/api/client.ts",
          name: "customInstance",
        },
        query: {
          useQuery: true,
          useMutation: true,
          signal: true,
        },
      },
    },
  },
});
