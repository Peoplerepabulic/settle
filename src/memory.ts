/**
 * memory.ts — per-space persistent memory.
 *
 * Stored as JSON at data/memory.json:
 *   - roster: senderId -> friendly label ("Player 1", "Player 2", …) so the
 *     agent can talk about voters without real names (Spectrum's User only
 *     carries an id).
 *   - preferences: learned tallies (cuisines, budgets) from past winners.
 *   - history: past decisions (topic, winner, timestamp).
 *
 * Writes are atomic (tmp file + rename) and happen on every mutation —
 * cheap at hackathon scale, safe against crashes.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

export interface DecisionRecord {
  topic: string;
  winner: string;
  at: string;
}

export interface SpaceMemory {
  roster: Record<string, string>;
  nextPlayer: number;
  preferences: {
    cuisines: Record<string, number>;
    budgets: Record<string, number>;
  };
  history: DecisionRecord[];
}

interface MemoryFile {
  spaces: Record<string, SpaceMemory>;
}

const DEFAULT_PATH = resolve(dirname(new URL(import.meta.url).pathname), "..", "data", "memory.json");

function blankSpace(): SpaceMemory {
  return { roster: {}, nextPlayer: 1, preferences: { cuisines: {}, budgets: {} }, history: [] };
}

export class MemoryStore {
  private path: string;
  private data: MemoryFile;

  private constructor(path: string, data: MemoryFile) {
    this.path = path;
    this.data = data;
  }

  static load(path: string = DEFAULT_PATH): MemoryStore {
    let data: MemoryFile = { spaces: {} };
    try {
      if (existsSync(path)) {
        const parsed = JSON.parse(readFileSync(path, "utf-8")) as Partial<MemoryFile>;
        if (parsed && typeof parsed.spaces === "object") data = { spaces: parsed.spaces };
      }
    } catch {
      // Corrupt file: start fresh rather than crash the agent.
      data = { spaces: {} };
    }
    return new MemoryStore(path, data);
  }

  hasSpace(spaceId: string): boolean {
    return spaceId in this.data.spaces;
  }

  getSpace(spaceId: string): SpaceMemory {
    if (!this.data.spaces[spaceId]) {
      this.data.spaces[spaceId] = blankSpace();
      this.save();
    }
    return this.data.spaces[spaceId];
  }

  /** Friendly label for a sender; assigns "Player N" on first sight. */
  labelFor(spaceId: string, senderId: string): string {
    const mem = this.getSpace(spaceId);
    if (!mem.roster[senderId]) {
      mem.roster[senderId] = `Player ${mem.nextPlayer++}`;
      this.save();
    }
    return mem.roster[senderId];
  }

  /** Record a finished decision and fold the winner's traits into preferences. */
  recordDecision(
    spaceId: string,
    rec: DecisionRecord,
    traits: { cuisine?: string; budget?: string },
  ): void {
    const mem = this.getSpace(spaceId);
    mem.history.push(rec);
    if (mem.history.length > 50) mem.history = mem.history.slice(-50);
    if (traits.cuisine) {
      const k = traits.cuisine.toLowerCase();
      mem.preferences.cuisines[k] = (mem.preferences.cuisines[k] ?? 0) + 1;
    }
    if (traits.budget) {
      const k = traits.budget.toLowerCase();
      mem.preferences.budgets[k] = (mem.preferences.budgets[k] ?? 0) + 1;
    }
    this.save();
  }

  /** Cuisines this group picked before, most-liked first. */
  likedCuisines(spaceId: string): string[] {
    const prefs = this.getSpace(spaceId).preferences.cuisines;
    return Object.entries(prefs)
      .sort((a, b) => b[1] - a[1])
      .map(([k]) => k)
      .slice(0, 5);
  }

  private save(): void {
    try {
      mkdirSync(dirname(this.path), { recursive: true });
      const tmp = this.path + ".tmp";
      writeFileSync(tmp, JSON.stringify(this.data, null, 2), "utf-8");
      renameSync(tmp, this.path);
    } catch {
      // Persistence is best-effort; never take the agent down.
    }
  }
}
