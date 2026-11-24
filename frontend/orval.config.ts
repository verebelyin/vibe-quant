// orval pinned exact: regen rewrites every generated file, so bump deliberately (bead vibe-quant-4sagi).
// Leave query.useQuery/useMutation unset: since 8.39 a global true forces it onto every verb
// (GETs became mutations); unset keeps the per-verb default (GET -> query, else -> mutation).
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
          signal: true,
        },
      },
    },
  },
});
