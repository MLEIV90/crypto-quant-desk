import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { applyThemeCssVariables } from "./theme";

// Antes de montar: las variables CSS del tema (ver theme.ts) tienen que
// estar disponibles desde el primer render de App.
applyThemeCssVariables();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
