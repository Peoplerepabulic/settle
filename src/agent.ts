/**
 * agent.ts — the Spectrum message loop.
 *
 * Owns the provider-agnostic send helpers (typing indicators, threaded
 * replies, best-effort reactions) and routes every inbound message to the
 * DecisionEngine. Anything provider-specific lives behind small guards so
 * the same code runs on iMessage and the terminal provider.
 */

import type { Message, Space } from "spectrum-ts";
import { DecisionEngine, EngineCtx } from "./decision.js";
import { LlmClient } from "./llm.js";
import { MemoryStore } from "./memory.js";

export interface AgentDeps {
  app: { messages: AsyncIterable<[Space, Message]> };
  llm: LlmClient;
  memory: MemoryStore;
  vetoWindowMs: number;
  tallyTimeoutMs: number;
}

/** Plain-text body of an inbound message ("" for non-text content). */
function messageText(message: Message): string {
  const c = message.content as { type?: string; text?: string; markdown?: string };
  if (c.type === "text" && typeof c.text === "string") return c.text;
  if (c.type === "markdown" && typeof c.markdown === "string") return c.markdown;
  return "";
}

interface ReactionInfo {
  emoji: string;
  targetMessageId?: string;
}

/** Platforms whose clients don't render Markdown (iMessage shows raw **). */
const PLAIN_TEXT_PLATFORMS = new Set(["imessage", "local_imessage"]);

/** Strip Markdown down to readable plain text for non-rendering clients. */
function stripMarkdown(text: string): string {
  return (
    text
      // links: [text](url) -> text (url)
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 ($2)")
      // bold / italic / strikethrough / inline code (longest markers first)
      .replace(/\*\*\*([^*]+)\*\*\*/g, "$1")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/__([^_]+)__/g, "$1")
      .replace(/~~([^~]+)~~/g, "$1")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/(^|\W)\*([^*\n]+)\*/g, "$1$2")
      .replace(/(^|\W)_([^_\n]+)_/g, "$1$2")
      // headings and blockquotes
      .replace(/^#{1,6}\s+/gm, "")
      .replace(/^>\s?/gm, "")
  );
}

function platformOf(space: Space, message?: Message): string {
  if (message?.platform) return message.platform;
  const p = (space as { __platform?: unknown }).__platform;
  return typeof p === "string" ? p : "";
}

/** Best-effort parse of an inbound reaction (tapback). */
function asReaction(message: Message): ReactionInfo | undefined {
  const c = message.content as {
    type?: string;
    emoji?: string;
    target?: { id?: string };
  };
  if (c.type !== "reaction" || typeof c.emoji !== "string") return undefined;
  return { emoji: c.emoji, targetMessageId: c.target?.id };
}

export async function runAgent(deps: AgentDeps): Promise<void> {
  const engine = new DecisionEngine(deps.llm, deps.memory, {
    vetoWindowMs: deps.vetoWindowMs,
    tallyTimeoutMs: deps.tallyTimeoutMs,
  });

  const ctx: EngineCtx = {
    async send(space: Space, text: string): Promise<string | undefined> {
      try {
        const body = PLAIN_TEXT_PLATFORMS.has(platformOf(space))
          ? stripMarkdown(text)
          : text;
        // responding() wraps the send in typing start/stop; providers
        // without typing indicators silently no-op.
        const sent = await space.responding(() => space.send(body));
        return sent?.id;
      } catch (err) {
        console.error("[settle] send failed:", err instanceof Error ? err.message : err);
        return undefined;
      }
    },
    async reply(message: Message, text: string): Promise<void> {
      try {
        const body = PLAIN_TEXT_PLATFORMS.has(platformOf(message.space, message))
          ? stripMarkdown(text)
          : text;
        await message.reply(body);
      } catch (err) {
        console.error("[settle] reply failed:", err instanceof Error ? err.message : err);
      }
    },
    async ack(message: Message, emoji: string): Promise<void> {
      try {
        // Resolves undefined where reactions are unsupported — that's fine.
        await message.react(emoji);
      } catch {
        // Best-effort only; never break the flow over a tapback.
      }
    },
  };

  console.log("[settle] listening for messages…");
  for await (const [space, message] of deps.app.messages) {
    try {
      // Ignore our own outbound traffic.
      if (message.direction === "outbound") continue;
      if (message.sender?.kind === "agent") continue;

      const senderId = message.sender?.id ?? "unknown";
      // Check before labelFor(), which creates the space entry.
      const isNewSpace = !deps.memory.hasSpace(space.id);

      const reaction = asReaction(message);
      if (reaction) {
        await engine.handleReaction(
          ctx, space, message, senderId,
          reaction.emoji, reaction.targetMessageId,
        );
        continue;
      }

      const text = messageText(message);
      if (!text.trim()) continue;

      if (isNewSpace) {
        await ctx.send(
          space,
          `👋 Hey! I'm **Settle** — the friend who actually makes the plan.\n\n` +
            `Add me to the chaos: debate dinner, weekend plans, movie night… ` +
            `I'll collect preferences, propose real options, run the vote, ` +
            `and even handle ties and vetoes.\n\n` +
            `Try: _"@Settle help us pick a dinner spot for Friday"_`,
        );
      }

      await engine.handleText(ctx, space, message, senderId, text);
    } catch (err) {
      console.error("[settle] handler error:", err instanceof Error ? err.message : err);
    }
  }
}
