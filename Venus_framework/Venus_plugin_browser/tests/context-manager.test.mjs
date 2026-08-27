import test from "node:test";
import assert from "node:assert/strict";

import {
  buildCompactionMessages,
  buildConversationContext,
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

test("compacts all raw history and leaves zero uncompressed entries", () => {
  const entries = Array.from({ length: 8 }, (_, index) => (
    entry(index + 1, "message", { role: "user", text: `task-${index + 1}` })
  ));
  const limits = {
    compactAfterPromptTokens: 6,
  };
  const plan = planCompaction({ summaryThrough: 0, promptTokens: 7 }, entries, limits);
  assert.equal(plan.throughSequence, 8);
  assert.equal(plan.entries.length, 8);
  assert.match(plan.transcript, /task-1/);
  assert.match(plan.transcript, /task-8/);
});

test("creates a focused chronological compaction request", () => {
  const plan = { transcript: "User task: find the price" };
  const messages = buildCompactionMessages({ summary: "Opened the store." }, plan);
  assert.equal(messages.length, 2);
  assert.match(messages[1].content, /Opened the store/);
  assert.match(messages[1].content, /find the price/);
  assert.match(messages[0].content, /用户让我查询了/);
  assert.match(messages[0].content, /按时间顺序/);
});

test("does not compact merely because the conversation has many short entries", () => {
  const entries = Array.from({ length: 100 }, (_, index) => (
    entry(index + 1, "message", { role: "user", text: "x" })
  ));
  const plan = planCompaction({ summaryThrough: 0, promptTokens: 1_000 }, entries, {
    compactAfterPromptTokens: 1_000,
  });
  assert.equal(plan, null);
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
