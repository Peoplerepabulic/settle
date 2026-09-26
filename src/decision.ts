/**
 * decision.ts — the group-decision state machine (one session per space).
 *
 *   idle → collecting → voting → veto → decided
 *                      ↘ (tie) runoff → voting (again)
 *
 * - collecting: gather constraints (cuisine, budget, location, dietary, when).
 * - voting:    numbered options; votes by number ("2", "i vote 2") or by
 *              reacting 1️⃣–4️⃣ on the options message. Votes can be changed.
 * - tie:       instant runoff among the tied leaders.
 * - veto:      "any vetoes?" window after a winner emerges; a veto drops the
 *              winner and re-runs the vote among the rest.
 * - decided:   winner announced with flair; preferences learned to memory.
 *
 * "@Settle cancel" (or just "cancel" mid-flow) aborts the session.
 * Late joiners are folded in automatically — anyone who votes gets a roster
 * label and is counted. Inactivity auto-tallies; everything persists to
 * data/memory.json via MemoryStore.
 */

import type { Message, Space } from "spectrum-ts";
import {
  Constraints,
  LlmClient,
  LlmUnavailableError,
  detectIntent,
  extractConstraints,
  generateCandidates,
} from "./llm.js";
import { MemoryStore } from "./memory.js";

/** What agent.ts must provide; keeps this module provider-agnostic. */
export interface EngineCtx {
  /** Send a message; resolves to the outbound message id (for reaction votes). */
  send(space: Space, text: string): Promise<string | undefined>;
  /** Threaded reply to a specific message. */
  reply(message: Message, text: string): Promise<void>;
  /** Best-effort emoji reaction (no-op where unsupported). */
  ack(message: Message, emoji: string): Promise<void>;
}

export interface Candidate {
  name: string;
  blurb: string;
}

export type Phase = "idle" | "collecting" | "voting" | "veto" | "decided";

interface Session {
  spaceId: string;
  phase: Phase;
  topic: string;
  constraints: Constraints;
  candidates: Candidate[];
  /** senderId -> candidate index */
  votes: Map<string, number>;
  /** outbound option-message ids that accept 1️⃣–4️⃣ reaction votes */
  optionMessageIds: Set<string>;
  decided?: Candidate;
  vetoTimer?: NodeJS.Timeout;
  tallyTimer?: NodeJS.Timeout;
  recent: Array<{ from: string; text: string }>;
  exchanges: number;
}

export interface EngineOptions {
  vetoWindowMs: number;
  tallyTimeoutMs: number;
}

const NUMBER_WORDS: Record<string, number> = {
  one: 1, two: 2, three: 3, four: 4, five: 5,
  first: 1, second: 2, third: 3, fourth: 4, fifth: 5,
};
const NUMBER_EMOJI: Record<string, number> = {
  "1️⃣": 1, "2️⃣": 2, "3️⃣": 3, "4️⃣": 4,
};

export class DecisionEngine {
  private sessions = new Map<string, Session>();
  private llm: LlmClient;
  private memory: MemoryStore;
  private opts: EngineOptions;

  constructor(llm: LlmClient, memory: MemoryStore, opts: EngineOptions) {
    this.llm = llm;
    this.memory = memory;
    this.opts = opts;
  }

  // ------------------------------------------------------------------ entry

  async handleText(
    ctx: EngineCtx,
    space: Space,
    message: Message,
    senderId: string,
    text: string,
  ): Promise<void> {
    const label = this.memory.labelFor(space.id, senderId);
    const session = this.sessions.get(space.id);
    const t = text.trim();

    if (this.isCancel(t)) {
      if (session && session.phase !== "idle" && session.phase !== "decided") {
        this.endSession(space.id);
        await ctx.send(space, "🗑️ Scrapped it. Shout when you need me — indecision is my cardio.");
      }
      return;
    }

    if (!session || session.phase === "idle" || session.phase === "decided") {
      await this.maybeStart(ctx, space, message, label, t);
      return;
    }

    this.note(session, label, t);
    this.pokeTallyTimer(ctx, space, session);

    switch (session.phase) {
      case "collecting":
        await this.onCollecting(ctx, space, session, t);
        break;
      case "voting":
        await this.onVoting(ctx, space, message, session, senderId, label, t);
        break;
      case "veto":
        await this.onVeto(ctx, space, session, label, t);
        break;
    }
  }

  async handleReaction(
    ctx: EngineCtx,
    space: Space,
    message: Message,
    senderId: string,
    emoji: string,
    targetMessageId: string | undefined,
  ): Promise<void> {
    const label = this.memory.labelFor(space.id, senderId);
    const session = this.sessions.get(space.id);
    if (!session) return;

    if (session.phase === "voting" && targetMessageId && session.optionMessageIds.has(targetMessageId)) {
      const n = NUMBER_EMOJI[emoji];
      if (n && n <= session.candidates.length) {
        session.votes.set(senderId, n - 1);
        this.pokeTallyTimer(ctx, space, session);
        await ctx.ack(message, "👍");
        await this.maybeAutoTally(ctx, space, session);
      }
      return;
    }

    if (session.phase === "veto" && (emoji === "🚫" || emoji === "👎")) {
      await this.applyVeto(ctx, space, session, label);
    }
  }

  // ------------------------------------------------------------------ start

  private async maybeStart(
    ctx: EngineCtx,
    space: Space,
    message: Message,
    label: string,
    text: string,
  ): Promise<void> {
    const lower = text.toLowerCase();

    // 1) Is this a group decision? (check first — "help us pick dinner"
    //    is a decision request, not a help request)
    let isDecision = /settle/.test(lower) && /\b(decide|decision|choose|pick|vote|where|what)\b/.test(lower);
    let topic = "";
    if (!isDecision) {
      try {
        const r = await detectIntent(this.llm, text);
        isDecision = r.isDecision;
        topic = r.topic;
      } catch (err) {
        if (err instanceof LlmUnavailableError) {
          const fb = this.fallbackIntent(text);
          isDecision = fb.isDecision;
          topic = fb.topic;
        } else {
          throw err;
        }
      }
    }

    // 2) Explicit help request (no decision detected).
    if (!isDecision && /\bhelp\b/.test(lower) && /settle/.test(lower)) {
      await ctx.reply(message, this.helpText());
      return;
    }

    if (!isDecision) {
      if (/settle/.test(lower)) {
        await ctx.reply(
          message,
          "👋 That's me! I'm Settle — I turn group-chat chaos into actual plans. " +
            "Try \"@Settle help us pick a dinner spot\" or just start debating and I'll jump in.",
        );
      }
      return;
    }

    const session: Session = {
      spaceId: space.id,
      phase: "collecting",
      topic: topic || this.guessTopic(text),
      constraints: {},
      candidates: [],
      votes: new Map(),
      optionMessageIds: new Set(),
      recent: [{ from: label, text }],
      exchanges: 0,
    };
    this.sessions.set(space.id, session);
    const liked = this.memory.likedCuisines(space.id);
    const nudge = liked.length ? ` (Psst — this crew has picked ${liked[0]} before 👀)` : "";
    await ctx.send(
      space,
      `🍽️ Say less — let's settle **${session.topic}**.${nudge}\n` +
        `Give me the vibe: cuisine? budget? area? any dietary needs?\n` +
        `_(Or just say "surprise us" and I'll propose.)_`,
    );
  }

  private helpText(): string {
    return (
      "🤝 **How I work**\n" +
      "1. Debate in the chat — I'll notice and start collecting preferences.\n" +
      "2. I'll propose 3–4 concrete options.\n" +
      "3. Vote by number (\"2\") or react 1️⃣–4️⃣ on my options message. Change your vote anytime.\n" +
      "4. Ties go to an instant runoff. Winner gets a veto round — speak now or forever hold your peace.\n" +
      `Commands: "tally" to count early, "cancel" to scrap it.`
    );
  }

  // -------------------------------------------------------------- collecting

  private async onCollecting(
    ctx: EngineCtx,
    space: Space,
    session: Session,
    text: string,
  ): Promise<void> {
    session.exchanges += 1;
    const lower = text.toLowerCase();
    const wantsProposal = /\b(propose|surprise us|just pick|go ahead|ready|done)\b/.test(lower);

    let merged: Constraints = { ...session.constraints };
    let ready = wantsProposal;
    try {
      const convo = session.recent.map((r) => `${r.from}: ${r.text}`).join("\n");
      const out = await extractConstraints(this.llm, session.topic, convo, session.constraints);
      merged = {
        cuisine: out.cuisine ?? session.constraints.cuisine,
        budget: out.budget ?? session.constraints.budget,
        location: out.location ?? session.constraints.location,
        dietary: out.dietary ?? session.constraints.dietary,
        when: out.when ?? session.constraints.when,
      };
      ready = ready || out.readyToPropose;
    } catch (err) {
      if (err instanceof LlmUnavailableError) {
        merged = this.fallbackConstraints(text, session.constraints);
        const known = Object.values(merged).filter(Boolean).length;
        ready = ready || known >= 2 || session.exchanges >= 3;
      } else {
        throw err;
      }
    }
    session.constraints = merged;

    if (!ready) {
      const missing = ["cuisine", "budget", "location"].filter(
        (k) => !merged[k as keyof Constraints],
      );
      await ctx.send(
        space,
        `Got it${merged.cuisine ? ` — **${merged.cuisine}** noted 🍜` : ""}` +
          `${merged.budget ? `, **${merged.budget}** budget` : ""}` +
          `${merged.dietary ? `, ${merged.dietary} friendly` : ""}. ` +
          (missing.length
            ? `Still curious about: ${missing.join(", ")}. Or say "propose" and I'll run with it.`
            : `Say "propose" whenever you're ready!`),
      );
      return;
    }
    await this.propose(ctx, space, session);
  }

  private async propose(ctx: EngineCtx, space: Space, session: Session): Promise<void> {
    session.phase = "voting";
    const liked = this.memory.likedCuisines(space.id);
    let candidates: Candidate[];
    try {
      candidates = await generateCandidates(this.llm, session.topic, session.constraints, liked);
    } catch (err) {
      if (err instanceof LlmUnavailableError) {
        candidates = this.fallbackCandidates(session.constraints);
      } else {
        throw err;
      }
    }
    if (!candidates.length) {
      session.phase = "collecting";
      await ctx.send(space, "Hmm, I blanked on options — give me one more hint? 🙈");
      return;
    }
    session.candidates = candidates;
    session.votes.clear();

    const lines = candidates.map((c, i) => `**${i + 1}. ${c.name}** — ${c.blurb}`);
    const msgId = await ctx.send(
      space,
      `🎯 Alright, here are my picks for **${session.topic}**:\n\n${lines.join("\n")}\n\n` +
        `Vote with the number, or react 1️⃣–4️⃣ on this message. ` +
        `Change your mind anytime. Say "tally" when everyone's in!`,
    );
    if (msgId) session.optionMessageIds.add(msgId);
    this.pokeTallyTimer(ctx, space, session);
  }

  // ----------------------------------------------------------------- voting

  private async onVoting(
    ctx: EngineCtx,
    space: Space,
    message: Message,
    session: Session,
    senderId: string,
    label: string,
    text: string,
  ): Promise<void> {
    const lower = text.toLowerCase();

    if (/\b(tally|results|count (the )?votes|that's everyone|everyone'?s in)\b/.test(lower)) {
      await this.tally(ctx, space, session);
      return;
    }

    const n = this.parseVote(lower, session.candidates.length);
    if (n !== undefined) {
      const prev = session.votes.get(senderId);
      session.votes.set(senderId, n - 1);
      this.pokeTallyTimer(ctx, space, session);
      await ctx.ack(message, "👍");
      if (prev !== undefined && prev !== n - 1) {
        await ctx.reply(message, `Switched you to **${n}. ${session.candidates[n - 1].name}** 🔄`);
      }
      await this.maybeAutoTally(ctx, space, session);
      return;
    }
    // Not a vote and not a command — stay quiet, keep the chat human.
  }

  /** If every known participant has voted, tally right away. */
  private async maybeAutoTally(ctx: EngineCtx, space: Space, session: Session): Promise<void> {
    const roster = this.memory.getSpace(session.spaceId).roster;
    const knownVoters = Object.keys(roster).length;
    if (knownVoters >= 2 && session.votes.size >= knownVoters) {
      await this.tally(ctx, space, session);
    }
  }

  private async tally(ctx: EngineCtx, space: Space, session: Session): Promise<void> {
    this.clearTallyTimer(session);
    if (session.votes.size === 0) {
      await ctx.send(space, "No votes yet! Drop a number 1–4 to get this show on the road 🗳️");
      this.pokeTallyTimer(ctx, space, session);
      return;
    }

    const counts = new Array<number>(session.candidates.length).fill(0);
    for (const idx of session.votes.values()) {
      if (idx >= 0 && idx < counts.length) counts[idx] += 1;
    }
    const best = Math.max(...counts);
    const leaders = counts
      .map((c, i) => ({ c, i }))
      .filter((x) => x.c === best)
      .map((x) => x.i);

    const scoreLine = session.candidates
      .map((c, i) => `${i + 1}. ${c.name} — ${counts[i]} vote${counts[i] === 1 ? "" : "s"}`)
      .join("\n");

    if (leaders.length > 1) {
      // Instant runoff among the tied leaders.
      session.votes.clear();
      const tiedNames = leaders.map((i) => `**${i + 1}. ${session.candidates[i].name}**`).join(", ");
      const msgId = await ctx.send(
        space,
        `📊 Tally:\n${scoreLine}\n\n😱 It's a tie between ${tiedNames}! ` +
          `Instant runoff — vote again, same numbers. May the best craving win.`,
      );
      if (msgId) session.optionMessageIds.add(msgId);
      this.pokeTallyTimer(ctx, space, session);
      return;
    }

    const winner = session.candidates[leaders[0]];
    session.phase = "veto";
    session.decided = winner;
    const msgId = await ctx.send(
      space,
      `📊 Tally:\n${scoreLine}\n\n🏆 Winner: **${winner.name}** — ${winner.blurb}\n\n` +
        `Any vetoes? Say "veto" or react 🚫 in the next ${Math.round(this.opts.vetoWindowMs / 1000)}s. ` +
        `Silence means it's locked in. 🔒`,
    );
    if (msgId) session.optionMessageIds.add(msgId);
    session.vetoTimer = setTimeout(() => {
      void this.finalize(ctx, space, session, false).catch(() => undefined);
    }, this.opts.vetoWindowMs);
    session.vetoTimer.unref?.();
  }

  // ------------------------------------------------------------------- veto

  private async onVeto(
    ctx: EngineCtx,
    space: Space,
    session: Session,
    label: string,
    text: string,
  ): Promise<void> {
    const lower = text.toLowerCase();
    // "no veto" contains the word veto — check the negation first.
    const saysVeto =
      (/\bveto\b/.test(lower) || lower.includes("🚫")) && !/\bno veto\b/.test(lower);
    if (saysVeto) {
      await this.applyVeto(ctx, space, session, label);
      return;
    }
    if (/\b(no veto|looks good|fine by me|ship it|let'?s go)\b/.test(lower)) {
      await this.finalize(ctx, space, session, false);
    }
    // Otherwise: silence is consent; the timer handles it.
  }

  private async applyVeto(ctx: EngineCtx, space: Space, session: Session, label: string): Promise<void> {
    this.clearVetoTimer(session);
    const out = session.decided;
    if (!out) return;
    session.candidates = session.candidates.filter((c) => c !== out);
    session.votes.clear();
    session.decided = undefined;

    if (session.candidates.length === 1) {
      session.decided = session.candidates[0];
      await this.finalize(ctx, space, session, true);
      return;
    }
    if (session.candidates.length === 0) {
      this.endSession(space.id);
      await ctx.send(space, "💀 Everything got vetoed. Bold. Let's start over — what's the vibe?");
      return;
    }
    session.phase = "voting";
    const lines = session.candidates.map((c, i) => `**${i + 1}. ${c.name}** — ${c.blurb}`);
    const msgId = await ctx.send(
      space,
      `🚫 ${label} vetoed **${out.name}**. Democracy is messy, I love it.\n\n` +
        `Re-vote among the survivors:\n${lines.join("\n")}\n\nNumber or 1️⃣–${["1️⃣", "2️⃣", "3️⃣", "4️⃣"][session.candidates.length - 1]} reaction — go!`,
    );
    if (msgId) session.optionMessageIds.add(msgId);
    this.pokeTallyTimer(ctx, space, session);
  }

  private async finalize(
    ctx: EngineCtx,
    space: Space,
    session: Session,
    afterVeto: boolean,
  ): Promise<void> {
    this.clearVetoTimer(session);
    this.clearTallyTimer(session);
    const winner = session.decided;
    if (!winner) {
      this.endSession(space.id);
      return;
    }
    session.phase = "decided";
    this.memory.recordDecision(
      space.id,
      { topic: session.topic, winner: winner.name, at: new Date().toISOString() },
      { cuisine: session.constraints.cuisine, budget: session.constraints.budget },
    );
    await ctx.send(
      space,
      `🎉 **IT'S DECIDED: ${winner.name}!** ${afterVeto ? "(survived a veto, respect)" : ""}\n` +
        `${winner.blurb} — enjoy! I'll remember this crew likes ${session.constraints.cuisine ?? "good food"} 😌`,
    );
    this.endSession(space.id);
  }

  // ---------------------------------------------------------------- helpers

  private endSession(spaceId: string): void {
    const s = this.sessions.get(spaceId);
    if (s) {
      this.clearVetoTimer(s);
      this.clearTallyTimer(s);
      this.sessions.delete(spaceId);
    }
  }

  private clearVetoTimer(s: Session): void {
    if (s.vetoTimer) {
      clearTimeout(s.vetoTimer);
      s.vetoTimer = undefined;
    }
  }

  private clearTallyTimer(s: Session): void {
    if (s.tallyTimer) {
      clearTimeout(s.tallyTimer);
      s.tallyTimer = undefined;
    }
  }

  private pokeTallyTimer(ctx: EngineCtx, space: Space, session: Session): void {
    this.clearTallyTimer(session);
    if (session.phase !== "voting") return;
    session.tallyTimer = setTimeout(() => {
      void this.tally(ctx, space, session).catch(() => undefined);
    }, this.opts.tallyTimeoutMs);
    session.tallyTimer.unref?.();
  }

  private note(session: Session, from: string, text: string): void {
    session.recent.push({ from, text });
    if (session.recent.length > 20) session.recent = session.recent.slice(-20);
  }

  private isCancel(text: string): boolean {
    const lower = text.toLowerCase();
    return (
      /\b(cancel|abort|never ?mind|scrap it|start over)\b/.test(lower) ||
      (/settle/.test(lower) && /\bstop\b/.test(lower))
    );
  }

  private guessTopic(text: string): string {
    const lower = text.toLowerCase();
    if (/\b(dinner|lunch|brunch|breakfast|eat|food|restaurant)\b/.test(lower)) return "dinner";
    if (/\b(movie|film|cinema)\b/.test(lower)) return "movie night";
    if (/\b(weekend|trip|hike|plans)\b/.test(lower)) return "weekend plans";
    return "the plan";
  }

  private parseVote(text: string, max: number): number | undefined {
    const t = text.toLowerCase().trim();
    // "2", "#2", "option 2", "number two", "i vote for 2", "change to 3"
    const digit = t.match(/(?:^|[\s#])(\d)(?:\s|$|[.,!])/);
    if (digit) {
      const n = parseInt(digit[1], 10);
      if (n >= 1 && n <= max) return n;
    }
    for (const [word, n] of Object.entries(NUMBER_WORDS)) {
      if (n <= max && new RegExp(`\\b${word}\\b`).test(t)) return n;
    }
    return undefined;
  }

  // ------------------------------------------------------- no-LLM fallbacks

  private fallbackIntent(text: string): { isDecision: boolean; topic: string } {
    const lower = text.toLowerCase();
    const mentionsSettle = /settle/.test(lower);
    const decisionish =
      /\b(where should we|what should we|decide|decision|choose|pick one|vote|options|suggestions?)\b/.test(lower);
    const isDecision = (mentionsSettle && decisionish) || /@settle/.test(lower) || /\bhelp us (pick|choose|decide)\b/.test(lower);
    return { isDecision, topic: isDecision ? this.guessTopic(text) : "" };
  }

  private fallbackConstraints(text: string, current: Constraints): Constraints {
    const lower = text.toLowerCase();
    const out = { ...current };
    const cuisines = [
      "ramen", "sushi", "japanese", "korean", "kbbq", "k-bbq", "bbq", "italian", "pizza",
      "mexican", "tacos", "thai", "chinese", "indian", "vietnamese", "pho",
      "burgers", "american", "mediterranean", "greek", "brunch", "seafood",
    ];
    for (const c of cuisines) {
      if (lower.includes(c)) {
        out.cuisine = c === "kbbq" || c === "k-bbq" ? "korean bbq" : c;
        break;
      }
    }
    if (/\$\$\$/.test(text)) out.budget = "$$$";
    else if (/\$\$/.test(text)) out.budget = "$$";
    else if (/\$(?!\d)/.test(text)) out.budget = "$";
    else if (/\b(cheap|budget|not too pricey|affordable)\b/.test(lower)) out.budget = "budget-friendly";
    else if (/\b(fancy|splurge|pricey|upscale)\b/.test(lower)) out.budget = "splurge";
    if (/\b(vegetarian|veggie|vegan|plant-based)\b/.test(lower)) out.dietary = "vegetarian-friendly";
    if (/\bhalal\b/.test(lower)) out.dietary = "halal";
    if (/\b(gluten[ -]?free)\b/.test(lower)) out.dietary = "gluten-free";
    const near = lower.match(/\bnear ([\w\s]+?)(?:[.,!?]|$)/);
    if (near) out.location = `near ${near[1].trim()}`;
    const when = lower.match(/\b(tonight|tomorrow|friday|saturday|sunday|this weekend|next week)\b/);
    if (when) out.when = when[1];
    return out;
  }

  private fallbackCandidates(c: Constraints): Candidate[] {
    const area = c.location ?? "nearby";
    const diet = c.dietary ? `, ${c.dietary}` : "";
    const q = (c.cuisine ?? "dinner").replace(/-/g, " ");
    return [
      { name: `Cozy ${q} spot ${area}`, blurb: `crowd-pleaser${diet}, easy parking` },
      { name: `Late-night ${q} joint`, blurb: `open late, big portions` },
      { name: `${q.charAt(0).toUpperCase() + q.slice(1)} house — local favorite`, blurb: `highly rated${diet}` },
      { name: `Budget ${q} pick`, blurb: `tasty and wallet-friendly` },
    ];
  }
}
