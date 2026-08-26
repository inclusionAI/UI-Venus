export const CONTEXT_LIMITS = Object.freeze({
  compactAfterChars: 48_000,
  compactAfterEntries: 60,
  keepRecentEntries: 24,
  requestContextChars: 36_000,
  compactionInputChars: 52_000,
});

export function planCompaction(conversation, entries, limits = CONTEXT_LIMITS) {
  const active = entries.filter(
    (entry) => entry.sequence > Number(conversation?.summaryThrough ?? 0),
  );
  const text = formatTranscript(active);
  if (
    active.length <= limits.compactAfterEntries
    && text.length <= limits.compactAfterChars
  ) {
    return null;
  }

  const compactCount = Math.max(1, active.length - limits.keepRecentEntries);
  const compactEntries = active.slice(0, compactCount);
  return {
    entries: compactEntries,
    throughSequence: compactEntries.at(-1).sequence,
    transcript: clipText(formatTranscript(compactEntries), limits.compactionInputChars, "start"),
  };
}

export function buildConversationContext(conversation, entries, limits = CONTEXT_LIMITS) {
  const sections = [];
  const summary = String(conversation?.summary ?? "").trim();
  if (summary) {
    sections.push(`Compacted conversation summary:\n${summary}`);
  }

  const active = entries.filter(
    (entry) => entry.sequence > Number(conversation?.summaryThrough ?? 0),
  );
  const recentTranscript = clipText(
    formatTranscript(active),
    limits.requestContextChars,
    "end",
  );
  if (recentTranscript) {
    sections.push(`Recent conversation transcript:\n${recentTranscript}`);
  }
  return sections.join("\n\n");
}

export function buildCompactionMessages(conversation, plan) {
  const previousSummary = String(conversation?.summary ?? "").trim() || "(none)";
  return [
    {
      role: "system",
      content: [
        "Summarize a browser-agent conversation for future context.",
        "Preserve user goals, completed work, important facts, current browser state, unresolved tasks, failures, and explicit preferences.",
        "Do not invent facts. Omit repetitive low-level mouse movements unless they matter.",
        "Return concise plain text only, without XML tags or an action.",
      ].join(" "),
    },
    {
      role: "user",
      content: `Previous summary:\n${previousSummary}\n\nTranscript to compact:\n${plan.transcript}`,
    },
  ];
}

export function estimateTokens(text) {
  return Math.ceil(String(text ?? "").length / 4);
}

export function formatTranscript(entries) {
  const lines = [];
  for (const entry of entries) {
    if (entry.kind === "message") {
      const label = entry.role === "user" ? "User task" : `Message (${entry.role ?? "system"})`;
      lines.push(`${label}: ${clipText(entry.text, 4000, "start")}`);
      const imageNames = (Array.isArray(entry.attachments) ? entry.attachments : [])
        .map((attachment) => String(attachment?.name || "image"))
        .slice(0, 8);
      if (imageNames.length) {
        lines.push(`Attached images: ${imageNames.join(", ")}`);
      }
      continue;
    }
    if (entry.kind === "step") {
      if (entry.think) lines.push(`Step ${entry.step} reasoning: ${clipText(entry.think, 1800, "start")}`);
      lines.push(`Step ${entry.step} action: ${entry.rawAction ?? entry.action?.raw ?? entry.action?.name ?? "unknown"}`);
      continue;
    }
    if (entry.kind === "result") {
      lines.push(`Step ${entry.step} result: ${clipText(safeJson(entry.result), 1800, "start")}`);
    }
  }
  return lines.join("\n");
}

function clipText(value, maxChars, keep) {
  const text = String(value ?? "").trim();
  if (text.length <= maxChars) return text;
  if (keep === "end") {
    return `[older content omitted]\n${text.slice(-maxChars)}`;
  }
  return `${text.slice(0, maxChars)}\n[newer content omitted]`;
}

function safeJson(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
