// dashboard/tailwind.config.ts
import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"DM Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "monospace"],
      },
      colors: {
        risk: {
          low: "#0d9488",
          high: "#dc2626",
          neutral: "#6b7280",
        },
        surface: {
          DEFAULT: "#ffffff",
          muted: "#f8fafc",
          inset: "#f1f5f9",
        },
        accent: {
          DEFAULT: "#0f172a",
          subtle: "#475569",
          muted: "#94a3b8",
        },
      },
      borderRadius: {
        card: "0.75rem",
      },
    },
  },
  plugins: [],
} satisfies Config;
