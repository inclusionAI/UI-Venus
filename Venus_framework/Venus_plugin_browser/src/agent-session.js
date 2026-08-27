import { ActionParseError, parseVenusResponse } from "./action-parser.js";

// Give the vision model a short visual timeline: the two preceding
// observations plus the current screenshot.
const HISTORY_IMAGE_WINDOW = 2;
const HISTORY_ROUND_WINDOW = 30;
const LOOP_REFLECTION = [
  "⚠️ REFLECTION: You have output the exact same response for 2 consecutive steps. ",
  "This indicates you are stuck in a loop. Carefully review the actions and SUMMARIZE the thoughts in previous steps. Analyze why your previous action did not change ",
  "the page state or why you are repeating the same action. ",
  "Consider: (1) Is the action actually executing? (2) Is the page unresponsive? ",
  "(3) Do you need to try a different approach? (4) Should you scroll to see more content? ",
  "AVOID repeating the same action - try something different to make progress.",
].join("");

export const AgentState = Object.freeze({
  IDLE: "idle",
  ATTACHING: "attaching",
  OBSERVING: "observing",
  THINKING: "thinking",
  EXECUTING: "executing",
  WAITING_USER: "waiting_user",
  FINISHED: "finished",
  STOPPED: "stopped",
  ERROR: "error",
});

export class AgentSession {
  constructor({
    browser,
    modelClient,
    promptTemplate,
    conversationContext = "",
    conversationImages = [],
    taskImages = [],
    previousCompletionTokens = 0,
    runId = globalThis.crypto?.randomUUID?.() ?? String(Date.now()),
    maxSteps = 100,
    onState = () => {},
    onStep = () => {},
    onThink = () => {},
    onFinal = () => {},
    onAttached = () => {},
    beforeAction = async () => {},
  }) {
    this.browser = browser;
    this.modelClient = modelClient;
    this.promptTemplate = promptTemplate;
    this.conversationContext = String(conversationContext ?? "").trim();
    this.conversationImages = normalizeImageAttachments(conversationImages);
    this.taskImages = normalizeImageAttachments(taskImages);
    this.previousCompletionTokens = Math.max(0, Number(previousCompletionTokens) || 0);
    this.runId = runId;
    this.maxSteps = maxSteps;
    this.onState = onState;
    this.onStep = onStep;
    this.onThink = onThink;
    this.onFinal = onFinal;
    this.onAttached = onAttached;
    this.beforeAction = beforeAction;
    this.state = AgentState.IDLE;
    this.running = false;
    this.runHistory = [];
    this.outputTokens = 0;
    this.hasOutputTokenUsage = false;
    this.promptTokens = 0;
    this.hasPromptTokenUsage = false;
    this.contextTokens = 0;
    this.abortController = null;
    this.releasePromise = null;
    this.task = "";
  }

  async run(task) {
    if (this.running) {
      throw new Error("已有任务正在运行");
    }
    this.task = String(task ?? "").trim();
    if (!this.task) {
      throw new Error("请输入任务");
    }

    this.running = true;
    this.runHistory = [];
    this.outputTokens = 0;
    this.hasOutputTokenUsage = false;
    this.promptTokens = 0;
    this.hasPromptTokenUsage = false;
    this.contextTokens = 0;
    this.abortController = new AbortController();
    this.releasePromise = null;
    let terminalState = AgentState.STOPPED;

    try {
      this.#setState(AgentState.ATTACHING, "正在接管当前标签页");
      const attachedTab = await this.browser.attachCurrentTab(this.task);
      this.onAttached(attachedTab);

      for (let step = 1; step <= this.maxSteps && this.running; step += 1) {
        this.#setState(AgentState.OBSERVING, `正在观察页面 · ${step}/${this.maxSteps}`);
        const observation = await this.browser.capture();

        this.#setState(AgentState.THINKING, `正在请求模型 · ${step}/${this.maxSteps}`);
        const parsed = await this.#requestValidAction(observation, step);
        const entry = {
          runId: this.runId,
          step,
          at: new Date().toISOString(),
          tab: observation.tab,
          viewport: observation.viewport,
          screenshot: observation.screenshot,
          think: parsed.think,
          rawResponse: parsed.rawResponse,
          rawAction: parsed.rawAction,
          action: parsed.action,
          result: null,
        };
        this.runHistory.push(entry);
        const staleScreenshot = this.runHistory.at(-(HISTORY_IMAGE_WINDOW + 1));
        if (staleScreenshot) delete staleScreenshot.screenshot;
        await this.onStep({ ...entry, phase: "proposed" });

        if (parsed.action.name === "finished") {
          terminalState = AgentState.FINISHED;
          entry.result = { action: "finished", content: parsed.action.content };
          await this.onStep({ ...entry, phase: "terminal" });
          await this.#releaseBrowser();
          await this.onFinal({
            type: "finished",
            content: parsed.action.content,
            step,
            outputTokens: this.hasOutputTokenUsage ? this.outputTokens : null,
            contextTokens: this.hasPromptTokenUsage ? this.contextTokens : null,
          });
          break;
        }

        if (parsed.action.name === "call_user") {
          terminalState = AgentState.WAITING_USER;
          entry.result = { action: "call_user", content: parsed.action.content };
          await this.onStep({ ...entry, phase: "terminal" });
          await this.#releaseBrowser();
          await this.onFinal({
            type: "call_user",
            content: parsed.action.content,
            step,
            outputTokens: this.hasOutputTokenUsage ? this.outputTokens : null,
            contextTokens: this.hasPromptTokenUsage ? this.contextTokens : null,
          });
          break;
        }

        this.#setState(AgentState.EXECUTING, `正在执行 ${parsed.rawAction}`);
        try {
          await this.beforeAction({
            action: parsed.action,
            observation,
            signal: this.abortController.signal,
          });
          entry.result = await this.browser.execute(parsed.action);
          await this.onStep({ ...entry, phase: "executed" });
        } catch (error) {
          if (this.abortController.signal.aborted || !this.running) {
            throw error;
          }
          entry.result = {
            action: parsed.action.name,
            ok: false,
            error: error instanceof Error ? error.message : String(error),
          };
          await this.onStep({ ...entry, phase: "failed" });
        }
      }

      if (this.running && terminalState === AgentState.STOPPED && this.runHistory.length >= this.maxSteps) {
        terminalState = AgentState.ERROR;
        throw new Error(`已达到最大步数 ${this.maxSteps}，任务尚未 Finished`);
      }
    } catch (error) {
      if (this.abortController.signal.aborted || !this.running) {
        terminalState = AgentState.STOPPED;
      } else {
        terminalState = AgentState.ERROR;
        throw error;
      }
    } finally {
      this.running = false;
      await this.#releaseBrowser();
      const labels = {
        [AgentState.FINISHED]: "任务完成，已释放浏览器",
        [AgentState.WAITING_USER]: "等待用户处理，已释放浏览器",
        [AgentState.STOPPED]: "任务已停止，已释放浏览器",
        [AgentState.ERROR]: "任务出错，已释放浏览器",
      };
      this.#setState(terminalState, labels[terminalState]);
    }
  }

  async stop() {
    if (!this.running) {
      await this.#releaseBrowser();
      return;
    }
    this.running = false;
    this.abortController?.abort(new Error("用户停止任务"));
    await this.#releaseBrowser();
  }

  #releaseBrowser() {
    if (!this.releasePromise) {
      this.releasePromise = Promise.resolve().then(() => this.browser.detach());
    }
    return this.releasePromise;
  }

  async #requestValidAction(observation, step) {
    const messages = this.#buildMessages(observation);
    let lastError = null;

    for (let attempt = 0; attempt < 2; attempt += 1) {
      const response = await this.modelClient.complete(messages, this.abortController.signal, {
        onProgress: ({ content, reasoningContent }) => {
          const think = extractStreamedThink(content, reasoningContent);
          if (think) this.onThink({ step, think });
        },
      });
      const outputTokens = Number(response?.usage?.completion_tokens ?? response?.usage?.output_tokens);
      if (Number.isFinite(outputTokens) && outputTokens >= 0) {
        this.outputTokens += outputTokens;
        this.hasOutputTokenUsage = true;
      }
      const promptTokens = Number(response?.usage?.prompt_tokens ?? response?.usage?.input_tokens);
      if (Number.isFinite(promptTokens) && promptTokens >= 0) {
        this.contextTokens = promptTokens;
        this.promptTokens = Math.max(this.promptTokens, promptTokens);
        this.hasPromptTokenUsage = true;
      }
      try {
        const parsed = parseVenusResponse(response.content);
        const reasoningThink = String(response?.reasoningContent ?? "").trim();
        if (!parsed.think && reasoningThink) {
          parsed.think = reasoningThink;
        }
        parsed.rawResponse = canonicalAssistantResponse(parsed.think, parsed.rawAction);
        const requiredAction = requiredActionForPendingTransfer(
          this.runHistory.at(-1)?.result,
          observation.fileTransfers,
        );
        if (requiredAction && parsed.action.name !== requiredAction) {
          throw new ActionParseError(
            `文件传输正在等待处理；下一个 action 必须是 ${actionDisplayName(requiredAction)}`,
            parsed.rawAction,
          );
        }
        if (
          parsed.action.name === "finished"
          && taskRequiresDownload(this.task)
          && !hasCompletedDownload(this.runHistory)
        ) {
          throw new ActionParseError(
            "下载任务尚未完成；只有 Download(...) 返回 status: completed 后才能 Finished",
            parsed.rawAction,
          );
        }
        return parsed;
      } catch (error) {
        lastError = error;
        if (!(error instanceof ActionParseError) || attempt === 1) {
          throw error;
        }
        messages.push({ role: "assistant", content: response.content });
        messages.push({
          role: "user",
          content: `Your previous response was invalid: ${error.message}. Return exactly one valid action wrapped in <action>...</action>.`,
        });
      }
    }
    throw lastError ?? new Error("无法解析模型 action");
  }

  #buildMessages(observation) {
    const prompt = this.promptTemplate
      .replaceAll("{current_date}", () => currentDate())
      .replaceAll("{task}", () => this.task);
    const messages = [{ role: "system", content: prompt }];

    if (this.conversationContext || this.conversationImages.length) {
      const content = [{
        type: "text",
        text: [
          "Context from earlier tasks in this same conversation follows.",
          "Use it as background, but rely on the current screenshot for live page state.",
          this.conversationContext,
          this.conversationImages.length
            ? `The user attached ${this.conversationImages.length} image(s) in recent earlier messages.`
            : "",
        ].filter(Boolean).join("\n\n"),
      }];
      appendImageParts(content, this.conversationImages);
      messages.push({
        role: "user",
        content,
      });
    }

    const recentHistory = this.runHistory.slice(-HISTORY_ROUND_WINDOW);
    const imageStart = Math.max(0, recentHistory.length - HISTORY_IMAGE_WINDOW);
    for (const [index, entry] of recentHistory.entries()) {
      const includeScreenshot = index >= imageStart && Boolean(entry.screenshot);
      const content = [
        { type: "text", text: historyObservationText(entry, includeScreenshot) },
      ];
      if (includeScreenshot) {
        content.push({
          type: "image_url",
          image_url: { url: `data:image/jpeg;base64,${entry.screenshot}` },
        });
      }
      messages.push({
        role: "user",
        content,
      });
      messages.push({ role: "assistant", content: entry.rawResponse });
    }

    const previousResult = this.runHistory.at(-1)?.result;
    const textParts = [
      `The URL of the current page is ${observation.tab.url}.`,
      `The current viewport is ${Math.round(observation.viewport.width)}x${Math.round(observation.viewport.height)} CSS pixels.`,
    ];
    if (previousResult) {
      const formattedResult = formatActionResult(previousResult);
      textParts.push(hasPendingFileTransfer(previousResult)
        ? formattedResult
        : `Result of the previous action:\n${formattedResult}`);
    }
    if (observation.fileTransfers?.length) {
      textParts.push(formatFileTransferEvents(observation.fileTransfers));
    }
    if (hasRepeatedResponse(this.runHistory)) {
      textParts.push(LOOP_REFLECTION);
    }
    const currentContent = [
      { type: "text", text: textParts.join("\n\n") },
    ];
    if (this.taskImages.length) {
      currentContent.push({
        type: "text",
        text: `User-provided reference images for the current task: ${this.taskImages.map((image) => image.name).join(", ")}`,
      });
      appendImageParts(currentContent, this.taskImages);
    }
    currentContent.push({ type: "text", text: "Current Screenshot:" });
    currentContent.push({
      type: "image_url",
      image_url: { url: `data:image/jpeg;base64,${observation.screenshot}` },
    });
    messages.push({
      role: "user",
      content: currentContent,
    });
    return messages;
  }

  #setState(state, label = "") {
    this.state = state;
    this.onState({ state, label });
  }
}

function extractStreamedThink(content, reasoningContent) {
  const reasoning = String(reasoningContent ?? "").trim();
  if (reasoning) return reasoning;
  const text = String(content ?? "");
  const start = text.indexOf("<think>");
  if (start < 0) return "";
  const streamed = text.slice(start + "<think>".length);
  const end = streamed.indexOf("</think>");
  return (end < 0 ? streamed : streamed.slice(0, end)).trim();
}

function canonicalAssistantResponse(think, rawAction) {
  return `<think>${String(think ?? "").trim()}</think>\n<action>${String(rawAction ?? "").trim()}</action>`;
}

function normalizeImageAttachments(images) {
  return (Array.isArray(images) ? images : [])
    .filter((image) => /^data:image\/(?:png|jpeg|webp|gif);base64,/i.test(String(image?.dataUrl || "")))
    .map((image) => ({
      name: String(image.name || "image"),
      dataUrl: String(image.dataUrl),
    }));
}

function hasRepeatedResponse(history) {
  const [previous, latest] = (Array.isArray(history) ? history : []).slice(-2);
  return Boolean(
    previous
    && latest
    && previous.think === latest.think
    && previous.rawAction === latest.rawAction,
  );
}

function appendImageParts(content, images) {
  for (const image of images) {
    content.push({
      type: "image_url",
      image_url: { url: image.dataUrl },
    });
  }
}

function historyObservationText(entry, includesScreenshot = false) {
  const lines = [
    `Previous observation at step ${entry.step}.`,
    `URL: ${entry.tab?.url ?? ""}`,
    includesScreenshot
      ? "The screenshot below is the observation used to choose this previous action."
      : "The previous screenshot is omitted; rely on the current screenshot for the live page state.",
  ];
  return lines.join("\n");
}

function formatActionResult(result) {
  const upload = result?.fileTransfers?.find(
    (event) => event?.type === "upload" && event?.status === "awaiting_selection",
  );
  if (upload) return formatUploadSelectionPrompt(upload);
  const download = result?.fileTransfers?.find(
    (event) => event?.type === "download" && event?.status === "awaiting_filename",
  );
  if (download) return formatDownloadNamingPrompt(download);
  const text = JSON.stringify(result, null, 2);
  return text.length <= 4000 ? text : `${text.slice(0, 4000)}\n...[truncated]`;
}

function requiredActionForPendingTransfer(previousResult, observationEvents) {
  const events = [
    ...(Array.isArray(previousResult?.fileTransfers) ? previousResult.fileTransfers : []),
    ...(Array.isArray(observationEvents) ? observationEvents : []),
  ];
  const pending = events.findLast?.((event) => (
    event?.type === "download" && event?.status === "awaiting_filename"
  ) || (
    event?.type === "upload" && event?.status === "awaiting_selection"
  )) ?? [...events].reverse().find((event) => (
    event?.type === "download" && event?.status === "awaiting_filename"
  ) || (
    event?.type === "upload" && event?.status === "awaiting_selection"
  ));
  if (pending?.type === "download") return "download";
  if (pending?.type === "upload") {
    return Array.isArray(pending.files) && pending.files.length === 0 ? "call_user" : "upload";
  }
  return "";
}

function actionDisplayName(action) {
  return {
    call_user: "CallUser(...)（workspace 中没有可上传文件）",
    download: "Download(filename='...')",
    upload: "Upload(file='...')",
  }[action] ?? action;
}

function taskRequiresDownload(task) {
  return /(?:下载|\bdownload\b)/i.test(String(task || ""));
}

function hasCompletedDownload(history) {
  return (Array.isArray(history) ? history : []).some((entry) => (
    entry?.result?.action === "download" && entry?.result?.status === "completed"
  ));
}

function hasPendingFileTransfer(result) {
  return result?.fileTransfers?.some(
    (event) => (
      event?.type === "upload" && event?.status === "awaiting_selection"
    ) || (
      event?.type === "download" && event?.status === "awaiting_filename"
    ),
  );
}

function formatFileTransferEvents(events) {
  const parts = [];
  for (const event of events) {
    if (event?.type === "upload" && event?.status === "awaiting_selection") {
      parts.push(formatUploadSelectionPrompt(event));
    } else if (event?.type === "download" && event?.status === "awaiting_filename") {
      parts.push(formatDownloadNamingPrompt(event));
    } else {
      parts.push(`File transfer event:\n${JSON.stringify(event, null, 2)}`);
    }
  }
  return parts.join("\n\n");
}

function formatDownloadNamingPrompt(event) {
  const lines = [
    "You've clicked a download button. The document is ready to be saved to the workspace.",
  ];
  if (event.linkText) lines.push(`Document: ${event.linkText}`);
  if (event.url) lines.push(`Source: ${event.url}`);
  if (event.suggestedFilename) lines.push(`Website-suggested name: ${event.suggestedFilename}`);
  lines.push("Choose a useful final filename based on the user's task, then use Download(filename='chosen-name.ext').");
  return lines.join("\n");
}

function formatUploadSelectionPrompt(event) {
  const files = Array.isArray(event.files) ? event.files : [];
  const lines = [
    "You've clicked an upload button. Here are the documents available in the workspace:",
  ];
  if (files.length === 0) {
    lines.push("- No documents are currently available.");
  } else {
    for (const file of files) {
      const details = [file.mimeType, formatFileSize(file.size)].filter(Boolean).join(", ");
      lines.push(`- ${file.path}${details ? ` (${details})` : ""}`);
    }
  }
  lines.push("Choose a document from this list, then use Upload(file='exact/relative/path').");
  return lines.join("\n");
}

function formatFileSize(value) {
  const bytes = Number(value);
  return Number.isFinite(bytes) && bytes >= 0 ? `${bytes} bytes` : "";
}

function currentDate() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}
