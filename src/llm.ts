/**
 * llm.ts — thin wrapper over any OpenAI-compatible chat completions API.
 *
 * All prompts ask for strict JSON so the decision engine can parse them
 * deterministically. If no LLM_API_KEY is set, every call throws
 * LlmUnavailableError and the engine falls back to keyword heuristics
 * (see decision.ts), so the terminal demo still runs end-to-end.
 */

export class LlmUnavailableError extends Error {
  constructor() {
    super("LLM not configured (set LLM_API_KEY)");
    this.name = "LlmUnavailableError";
  }
}

export interface LlmConfig {
  apiKey: string;
  baseUrl: string;
  model: string;
}

interface ChatMessage {
  role: "system" | "user";
  content: string;
}

export class LlmClient {
  readonly available: boolean;
  private cfg: LlmConfig;

  constructor(cfg: LlmConfig) {
    this.cfg = cfg;
    this.available = cfg.apiKey.length > 0;
  }

  /** Require the LLM; throws LlmUnavailableError when no key is configured. */
  private require(): LlmConfig {
    if (!this.available) throw new LlmUnavailableError();
    return this.cfg;
  }

  private async post(messages: ChatMessage[], jsonMode: boolean): Promise<string> {
    const cfg = this.require();
    const url = cfg.baseUrl.replace(/\/$/, "") + "/chat/completions";
    const body: Record<string, unknown> = {
      model: cfg.model,
      messages,
      temperature: 0.4,
      max_tokens: 800,
    };
    if (jsonMode) body.response_format = { type: "json_object" };

    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${cfg.apiKey}`,
        },
        body: JSON.stringify(body),
      });
    } catch (err) {
      throw new Error(`LLM request failed: ${err instanceof Error ? err.message : String(err)}`);
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`LLM API error ${res.status}: ${text.slice(0, 300)}`);
    }
    const data = (await res.json()) as {
      choices?: Array<{ message?: { content?: string | null } }>;
    };
    const content = data.choices?.[0]?.message?.content;
    if (!content) throw new Error("LLM returned an empty response");
    return content;
  }

  /** Chat completion parsed as JSON. Strips markdown fences if present. */
  async json<T>(system: string, user: string): Promise<T> {
    const raw = await this.post(
      [
        { role: "system", content: system + "\nReturn ONLY valid JSON, no markdown fences." },
        { role: "user", content: user },
      ],
      true,
    );
    const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
    try {
      return JSON.parse(cleaned) as T;
    } catch {
      throw new Error(`LLM returned invalid JSON: ${cleaned.slice(0, 200)}`);
    }
  }

  /** Plain-text chat completion (for short phrasing). */
  async text(system: string, user: string): Promise<string> {
    return (await this.post(
      [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      false,
    )).trim();
  }
}

// ---------------------------------------------------------------------------
// Prompt builders — kept here so decision.ts stays focused on the state machine
// ---------------------------------------------------------------------------

export interface IntentResult {
  isDecision: boolean;
  topic: string;
}

/** Shared shape for decision constraints (used by llm.ts and decision.ts). */
export interface Constraints {
  cuisine?: string;
  budget?: string;
  location?: string;
  dietary?: string;
  when?: string;
}

const INTENT_SYSTEM = `You read one message from a group chat and decide whether the
group is trying to make a GROUP DECISION (where to eat, what to do this weekend,
which movie, etc.). A casual statement ("I'm having pizza tonight") is NOT a
decision. A question or plea aimed at the group ("where should we eat friday?",
"someone pick a movie", "we need to decide") IS a decision.
Return JSON: {"isDecision": boolean, "topic": string}
topic is a short noun phrase like "dinner", "weekend plans", "movie night".
If isDecision is false, topic may be "".`;

export async function detectIntent(llm: LlmClient, message: string): Promise<IntentResult> {
  const out = await llm.json<IntentResult>(INTENT_SYSTEM, `Message: """${message}"""`);
  return {
    isDecision: out.isDecision === true,
    topic: typeof out.topic === "string" ? out.topic.slice(0, 60) : "",
  };
}

export interface ConstraintsResult {
  cuisine?: string;
  budget?: string;
  location?: string;
  dietary?: string;
  when?: string;
  readyToPropose: boolean;
}

const CONSTRAINTS_SYSTEM = `You extract decision constraints from a group chat.
You are given the conversation so far and the constraints collected already.
Merge in anything new, keep old values unless contradicted.
Return JSON: {"cuisine": string|null, "budget": string|null, "location": string|null,
"dietary": string|null, "when": string|null, "readyToPropose": boolean}
- cuisine: e.g. "ramen", "korean bbq". budget: "$", "$$", "$$$", or words like "cheap".
- location: area/neighborhood. dietary: "vegetarian options", "halal", etc.
- when: "friday night", "sunday afternoon", etc.
- readyToPropose: true if we have enough to suggest concrete options (cuisine OR
  budget OR location known, or the group explicitly said "just propose" / "surprise us").
Use null for unknown fields.`;

export async function extractConstraints(
  llm: LlmClient,
  topic: string,
  conversation: string,
  current: Constraints,
): Promise<ConstraintsResult> {
  const out = await llm.json<ConstraintsResult>(
    CONSTRAINTS_SYSTEM,
    `Topic: ${topic}\nCurrent constraints: ${JSON.stringify(current)}\n` +
      `Conversation:\n${conversation}`,
  );
  const clean = (v: unknown): string | undefined =>
    typeof v === "string" && v.trim() ? v.trim().slice(0, 80) : undefined;
  return {
    cuisine: clean(out.cuisine) ?? current.cuisine,
    budget: clean(out.budget) ?? current.budget,
    location: clean(out.location) ?? current.location,
    dietary: clean(out.dietary) ?? current.dietary,
    when: clean(out.when) ?? current.when,
    readyToPropose: out.readyToPropose === true,
  };
}

export interface CandidateResult {
  name: string;
  blurb: string;
}

const CANDIDATES_SYSTEM = `You suggest concrete options for a group's decision.
Return JSON: {"candidates": [{"name": string, "blurb": string}, ...]}
- Suggest exactly 4 options. Names should be realistic, specific places or plans
  (use your knowledge of the area if a location is given).
- blurb: one short phrase, max 12 words, why it fits (e.g. "late-night ramen, veggie broth available").
- Vary them: mix cuisines/styles, include at least one budget-friendly pick.
- Respect dietary constraints strictly.`;

export async function generateCandidates(
  llm: LlmClient,
  topic: string,
  constraints: Constraints,
  likedBefore: string[],
): Promise<CandidateResult[]> {
  const out = await llm.json<{ candidates: CandidateResult[] }>(
    CANDIDATES_SYSTEM,
    `Topic: ${topic}\nConstraints: ${JSON.stringify(constraints)}\n` +
      `This group liked these in the past: ${likedBefore.join(", ") || "(nothing yet)"}`,
  );
  const list = Array.isArray(out.candidates) ? out.candidates : [];
  return list
    .filter((c) => typeof c?.name === "string" && c.name.trim())
    .slice(0, 4)
    .map((c) => ({
      name: c.name.trim().slice(0, 60),
      blurb: typeof c.blurb === "string" ? c.blurb.trim().slice(0, 90) : "",
    }));
}
