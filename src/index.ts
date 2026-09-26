/**
 * index.ts — composition root.
 *
 *   PROVIDER=terminal  → local CLI chat, no credentials (demo/dev)
 *   PROVIDER=imessage  → Photon cloud iMessage (needs dashboard credentials)
 *
 * No secrets live in code — everything comes from the environment.
 */

import { Spectrum } from "spectrum-ts";
import { imessage } from "spectrum-ts/providers/imessage";
import { terminal } from "spectrum-ts/providers/terminal";
import { runAgent } from "./agent.js";
import { LlmClient } from "./llm.js";
import { MemoryStore } from "./memory.js";

function env(name: string, fallback = ""): string {
  return process.env[name] ?? fallback;
}

function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  const n = raw ? parseInt(raw, 10) : NaN;
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

async function main(): Promise<void> {
  const provider = env("PROVIDER", "imessage").toLowerCase();

  // The two Spectrum() overloads return different instance types; `any`
  // keeps the composition root simple since runAgent only needs .messages.
  let app: any;
  if (provider === "terminal") {
    console.log("[settle] provider=terminal (local demo, no credentials needed)");
    app = await Spectrum({ providers: [terminal.config()] });
  } else if (provider === "imessage") {
    const projectId = env("SPECTRUM_PROJECT_ID");
    const projectSecret = env("SPECTRUM_PROJECT_SECRET");
    if (!projectId || !projectSecret) {
      console.error(
        "[settle] PROVIDER=imessage needs SPECTRUM_PROJECT_ID and SPECTRUM_PROJECT_SECRET.\n" +
          "Get them from the Photon dashboard (https://app.photon.codes), or run with PROVIDER=terminal.",
      );
      process.exit(1);
    }
    console.log("[settle] provider=imessage (Photon cloud)");
    app = await Spectrum({
      projectId,
      projectSecret,
      providers: [imessage.config()],
    });
  } else {
    console.error(`[settle] unknown PROVIDER="${provider}" (expected "imessage" or "terminal")`);
    process.exit(1);
  }

  const llm = new LlmClient({
    apiKey: env("LLM_API_KEY"),
    baseUrl: env("LLM_BASE_URL", "https://api.openai.com/v1"),
    model: env("LLM_MODEL", "gpt-4o-mini"),
  });
  console.log(
    `[settle] llm=${llm.available ? env("LLM_MODEL", "gpt-4o-mini") : "unavailable — using keyword heuristics"}`,
  );

  const memory = MemoryStore.load();

  process.on("SIGINT", () => {
    console.log("\n[settle] shutting down…");
    process.exit(0);
  });

  await runAgent({
    app,
    llm,
    memory,
    vetoWindowMs: envInt("VETO_WINDOW_MS", 60_000),
    tallyTimeoutMs: envInt("TALLY_TIMEOUT_MS", 600_000),
  });
}

main().catch((err) => {
  console.error("[settle] fatal:", err instanceof Error ? err.message : err);
  process.exit(1);
});
