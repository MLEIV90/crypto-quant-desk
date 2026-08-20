import { defineConfig } from "vitest/config";

// Config separada de vite.config.ts (Fase 14, hallazgo F-02) a propósito:
// así `npm run build`/`npm run dev` no se ven afectados por nada de esto,
// y `npm run test` solo corre los tests unitarios de lógica pura (ver
// src/**/*.test.ts) sin depender de un browser real.
export default defineConfig({
  test: {
    environment: "jsdom",
    globals: false,
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
