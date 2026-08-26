import test from "node:test";
import assert from "node:assert/strict";

import {
  buildCompactionMessages,
  buildConversationContext,
  estimateTokens,
  formatTranscript,
  planCompaction,
} from "../src/context-manager.js";

function entry(sequence, kind, extra = {}) {
  return { sequence, kind, ...extra };
}

test("builds context from summary and entries after the compacted boundary", () => {
  const conversation = { summary: "User is comparing cameras.", summaryThrough: 2 };
  const context = buildConversationContext(conversation, [
    entry(1, "message", { role: "user", text: "old task" }),
    entry(2, "result", { step: 1, result: { action: "click" } }),
    entry(3, "message", { role: "user", text: "continue with the cheaper model" }),
    entry(4, "step", { step: 1, rawAction: "Click(point=(10,20))" }),
  ]);

  assert.match(context, /User is comparing cameras/);
  assert.doesNotMatch(context, /old task/);
  assert.match(context, /continue with the cheaper model/);
  assert.match(context, /Click\(point=\(10,20\)\)/);
});

test("plans compaction while retaining recent entries", () => {
  const entries = Array.from({ length: 8 }, (_, index) => (
    entry(index + 1, "message", { role: "user", text: `task-${index + 1}` })
  ));
  const limits = {
    compactAfterChars: 1_000_000,
    compactAfterEntries: 6,
    keepRecentEntries: 3,
    requestContextChars: 1_000,
    compactionInputChars: 1_000,
  };
  const plan = planCompaction({ summaryThrough: 0 }, entries, limits);
  assert.equal(plan.throughSequence, 5);
  assert.equal(plan.entries.length, 5);
  assert.match(plan.transcript, /task-1/);
  assert.doesNotMatch(plan.transcript, /task-8/);
});

test("creates a focused compaction request and estimates tokens", () => {
  const plan = { transcript: "User task: find the price" };
  const messages = buildCompactionMessages({ summary: "Opened the store." }, plan);
  assert.equal(messages.length, 2);
  assert.match(messages[1].content, /Opened the store/);
  assert.match(messages[1].content, /find the price/);
  assert.equal(estimateTokens("12345678"), 2);
});

test("includes image names without embedding image data in transcript", () => {
  const transcript = formatTranscript([
    entry(1, "message", {
      role: "user",
      text: "compare this",
      attachments: [{ name: "reference.png", dataUrl: "data:image/png;base64,secret" }],
    }),
  ]);

  assert.match(transcript, /Attached images: reference\.png/);
  assert.doesNotMatch(transcript, /base64/);
});
