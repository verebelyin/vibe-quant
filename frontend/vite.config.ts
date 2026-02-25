import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      // @ts-expect-error -- import.meta.dirname is available in Node 21+ / Vite 7
      "@": `${import.meta.dirname}/src`,
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes("node_modules/react-dom/") || id.includes("node_modules/react/")) {
            return "vendor-react";
          }
          if (id.includes("node_modules/recharts/") || id.includes("node_modules/d3-")) {
            return "vendor-recharts";
          }
          if (id.includes("node_modules/radix-ui/")) {
            return "vendor-radix";
          }
          if (
            id.includes("node_modules/@tanstack/react-router/") ||
            id.includes("node_modules/@tanstack/react-query/")
          ) {
            return "vendor-tanstack";
          }
          if (
            id.includes("node_modules/plotly.js-dist-min/") ||
            id.includes("node_modules/react-plotly.js/")
          ) {
            return "vendor-plotly";
          }
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
