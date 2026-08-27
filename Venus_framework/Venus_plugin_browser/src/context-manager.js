export const CONTEXT_LIMITS = Object.freeze({
  compactAfterPromptTokens: 32_000,
});

export function planCompaction(conversation, entries, limits = CONTEXT_LIMITS) {
  const active = entries.filter(
    (entry) => entry.sequence > Number(conversation?.summaryThrough ?? 0),
  );
  const promptTokens = Number(conversation?.promptTokens);
  if (!Number.isFinite(promptTokens) || promptTokens <= limits.compactAfterPromptTokens) {
    return null;
  }
  if (active.length === 0) return null;

  const compactEntries = active;
  return {
    entries: compactEntries,
    throughSequence: compactEntries.at(-1).sequence,
    transcript: formatTranscript(compactEntries),
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
  const recentTranscript = formatTranscript(active);
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
        "请将浏览器代理对话压缩为简洁、按时间顺序的中文任务摘要。",
        "使用类似“用户让我查询了……，结果是……；用户继续要求……，我进一步给出结果……”的叙述形式。",
        "重点保留用户目标、关键条件、最终结果、后续要求、尚未解决事项和明确偏好。",
        "合并连续的低层点击、滚动和等待动作，除非它们与失败原因或最终结果直接相关。",
        "不要虚构信息，不要输出思考过程、XML 标签或 action，只返回摘要正文。",
      ].join(" "),
    },
    {
      role: "user",
      content: `Previous summary:\n${previousSummary}\n\nTranscript to compact:\n${plan.transcript}`,
    },
  ];
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
