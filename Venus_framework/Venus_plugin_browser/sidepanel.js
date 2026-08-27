/**
 * 使用示例：可在 Side Panel 或网页底部的页面助手输入任务，按 Ctrl/⌘ + Enter 或点击箭头运行。
 */
import { AgentSession, AgentState } from "./src/agent-session.js";
import { BrowserBridge } from "./src/browser-bridge.js";
import { ConversationStore } from "./src/conversation-store.js";
import {
  missingConfigurationFields,
  missingConfigurationMessage,
} from "./src/config-validation.js";
import {
  buildCompactionMessages,
  buildConversationContext,
  planCompaction,
} from "./src/context-manager.js";
import { OpenAICompatibleClient } from "./src/model-client.js";
import {
  hasApiPermission,
  loadSettings,
  requestApiPermission,
  saveSettings,
} from "./src/settings.js";
import {
  loadWorkspaceHandle,
  saveWorkspaceHandle,
  verifyWorkspacePermission,
} from "./src/workspace-store.js";

const elements = {
  settingsToggle: byId("settings-toggle"),
  controlPanelToggle: byId("control-panel-toggle"),
  settingsClose: byId("settings-close"),
  settingsPanel: byId("settings-panel"),
  apiUrl: byId("api-url"),
  model: byId("model"),
  maxSteps: byId("max-steps"),
  temperature: byId("temperature"),
  apiKey: byId("api-key"),
  rememberKey: byId("remember-key"),
  workspaceLabel: byId("workspace-label"),
  chooseWorkspace: byId("choose-workspace"),
  keyToggle: byId("key-toggle"),
  testModel: byId("test-model"),
  saveSettings: byId("save-settings"),
  settingsFeedback: byId("settings-feedback"),
  statusDot: byId("status-dot"),
  statusLabel: byId("status-label"),
  tabLabel: byId("tab-label"),
  conversationSelect: byId("conversation-select"),
  deleteConversation: byId("delete-conversation"),
  conversation: byId("conversation"),
  emptyState: byId("empty-state"),
  messages: byId("messages"),
  composer: byId("composer"),
  taskInput: byId("task-input"),
  imageInput: byId("image-input"),
  imagePreview: byId("image-preview"),
  addImage: byId("add-image"),
  runTask: byId("run-task"),
  stopTask: byId("stop-task"),
  clearChat: byId("clear-chat"),
};

const browser = new BrowserBridge();
const conversationStore = new ConversationStore();
const stepCards = new Map();
let promptTemplate = "";
let activeSession = null;
let taskRunLocked = false;
let currentSettings = null;
let currentConversation = null;
let workspaceHandle = null;
let pendingImages = [];
let controlHeartbeatTimer = null;
let panelInitialized = false;
let deferredPageTaskTabId = null;
const workspaceNoticeConversations = new Set();

const IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);
const MAX_IMAGES = 4;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_IMAGE_TOTAL_BYTES = 20 * 1024 * 1024;
const CONTROL_PANEL_ENABLED_KEY = "controlPanelEnabled";
const HAS_NATIVE_SIDE_PANEL_VISIBILITY = Boolean(
  chrome.sidePanel?.onOpened && chrome.sidePanel?.onClosed,
);

initialize().catch((error) => {
  renderError(`插件初始化失败：${error.message}`);
  setStatus(AgentState.ERROR, "初始化失败");
});

async function initialize() {
  browser.connect();
  browser.onEvent((event, payload) => {
    if (event === "debugger_detached" && activeSession?.running) {
      if (payload.recoverable) {
        setStatus(AgentState.OBSERVING, "页面目标已切换，正在重新接管");
      } else {
        stopControlHeartbeat();
        const message = `浏览器控制已被释放：${payload.reason ?? "unknown"}`;
        renderError(message);
        stopTask(message).catch(handleUiError);
      }
    } else if (event === "disconnect" && activeSession?.running) {
      stopControlHeartbeat();
      stopTask(payload.reason ?? "插件后台连接已断开，任务已终止。").catch(handleUiError);
    } else if (event === "stop_requested" && activeSession?.running) {
      stopTask().catch(handleUiError);
    } else if (event === "page_task_available") {
      deferredPageTaskTabId = payload.tabId;
      if (panelInitialized) drainPageTaskInbox().catch(handleUiError);
    } else if (event === "file_transfer") {
      renderFileTransferEvent(payload);
    }
  });

  const [loadedPrompt, loadedSettings, panelPreference] = await Promise.all([
    fetch(chrome.runtime.getURL("prompts/venus_system.txt")).then((response) => {
      if (!response.ok) throw new Error("无法加载 Venus prompt");
      return response.text();
    }),
    loadSettings(),
    chrome.storage.local.get([CONTROL_PANEL_ENABLED_KEY]),
  ]);
  promptTemplate = loadedPrompt;
  currentSettings = loadedSettings;
  setControlPanelToggle(panelPreference[CONTROL_PANEL_ENABLED_KEY] !== false);

  populateSettings(currentSettings);
  elements.settingsPanel.classList.toggle(
    "collapsed",
    Boolean(currentSettings.apiUrl && currentSettings.model && currentSettings.apiKey),
  );
  currentConversation = await conversationStore.getOrCreateActiveConversation();
  await refreshConversationList();
  await restoreConversation(currentConversation.id);
  await refreshWorkspaceUi();
  bindEvents();
  setStatus(AgentState.IDLE, "空闲");
  if (controlPanelToggleEnabled()) {
    await browser.showControl().catch((error) => {
      console.warn("Venus could not show the idle control panel", error);
    });
  } else {
    await browser.setControlEnabled(false);
  }
  panelInitialized = true;
  await drainPageTaskInbox();
  syncPanelVisibility();
}

async function drainPageTaskInbox() {
  if (!panelInitialized || isTaskBusy()) return;
  const tabId = deferredPageTaskTabId;
  deferredPageTaskTabId = null;
  const claimed = await browser.claimPageTask(tabId).catch(() => null);
  const pendingTask = claimed?.pendingTask;
  if (!pendingTask?.task || isTaskBusy()) return;
  elements.taskInput.value = pendingTask.task;
  await runTask({ task: pendingTask.task, source: "control" });
}

function bindEvents() {
  elements.settingsToggle.addEventListener("click", () => {
    elements.settingsPanel.classList.toggle("collapsed");
  });
  elements.settingsClose.addEventListener("click", () => {
    elements.settingsPanel.classList.add("collapsed");
  });
  elements.controlPanelToggle.addEventListener("click", () => {
    const enabled = !controlPanelToggleEnabled();
    setControlPanelToggle(enabled);
    setControlPanelEnabled(enabled).catch(handleUiError);
  });
  elements.keyToggle.addEventListener("click", () => {
    const reveal = elements.apiKey.type === "password";
    elements.apiKey.type = reveal ? "text" : "password";
    elements.keyToggle.textContent = reveal ? "隐藏" : "显示";
  });
  elements.saveSettings.addEventListener("click", saveSettingsFromForm);
  elements.testModel.addEventListener("click", testModelConnection);
  elements.chooseWorkspace.addEventListener("click", () => chooseWorkspace().catch(handleUiError));
  elements.addImage.addEventListener("click", () => elements.imageInput.click());
  elements.imageInput.addEventListener("change", () => {
    addImageFiles(elements.imageInput.files).catch(handleUiError);
    elements.imageInput.value = "";
  });
  elements.runTask.addEventListener("click", () => runTask().catch(handleUiError));
  elements.stopTask.addEventListener("click", () => stopTask().catch(handleUiError));
  elements.clearChat.addEventListener("click", () => startNewConversation().catch(handleUiError));
  elements.deleteConversation.addEventListener("click", () => deleteCurrentConversation().catch(handleUiError));
  elements.conversationSelect.addEventListener("change", () => switchConversation().catch(handleUiError));
  elements.taskInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      runTask().catch(handleUiError);
    }
  });
  elements.taskInput.addEventListener("paste", (event) => {
    const images = [...(event.clipboardData?.files ?? [])].filter((file) => file.type.startsWith("image/"));
    if (!images.length) return;
    event.preventDefault();
    addImageFiles(images).catch(handleUiError);
  });
  elements.composer.addEventListener("dragover", (event) => {
    if (isTaskBusy()) return;
    event.preventDefault();
    elements.composer.classList.add("dragging");
  });
  elements.composer.addEventListener("dragleave", () => elements.composer.classList.remove("dragging"));
  elements.composer.addEventListener("drop", (event) => {
    elements.composer.classList.remove("dragging");
    if (isTaskBusy()) return;
    event.preventDefault();
    addImageFiles(event.dataTransfer?.files).catch(handleUiError);
  });
  window.addEventListener("pagehide", () => {
    if (!HAS_NATIVE_SIDE_PANEL_VISIBILITY) {
      browser.setPanelVisibility(false).catch(() => {});
    }
    stopControlHeartbeat();
    activeSession?.stop();
  });
  if (!HAS_NATIVE_SIDE_PANEL_VISIBILITY) {
    document.addEventListener("visibilitychange", syncPanelVisibility);
  }
}

function syncPanelVisibility() {
  if (HAS_NATIVE_SIDE_PANEL_VISIBILITY) return;
  browser.setPanelVisibility(document.visibilityState === "visible").catch(() => {});
}

async function setControlPanelEnabled(enabled) {
  elements.controlPanelToggle.disabled = true;
  try {
    await chrome.storage.local.set({ [CONTROL_PANEL_ENABLED_KEY]: enabled });
    await browser.setControlEnabled(enabled);
  } catch (error) {
    setControlPanelToggle(!enabled);
    await chrome.storage.local.set({ [CONTROL_PANEL_ENABLED_KEY]: !enabled }).catch(() => {});
    throw error;
  } finally {
    elements.controlPanelToggle.disabled = false;
  }
}

function controlPanelToggleEnabled() {
  return elements.controlPanelToggle.getAttribute("aria-checked") === "true";
}

function setControlPanelToggle(enabled) {
  elements.controlPanelToggle.setAttribute("aria-checked", enabled ? "true" : "false");
  elements.controlPanelToggle.setAttribute("aria-label", enabled ? "隐藏页面助手" : "显示页面助手");
}

async function saveSettingsFromForm() {
  setSettingsBusy(true);
  feedback("正在检查设置……");
  try {
    const draft = settingsFromForm();
    const missing = await validateRequiredConfiguration(draft);
    if (missing.length) {
      throw new Error(missingConfigurationMessage(missing));
    }
    feedback("正在请求模型地址访问权限……");
    const granted = await requestApiPermission(draft.apiUrl);
    if (!granted) {
      throw new Error("未授予模型 API 地址访问权限");
    }
    currentSettings = await saveSettings(draft);
    populateSettings(currentSettings);
    feedback("设置已保存", "success");
  } catch (error) {
    feedback(error.message, "error");
  } finally {
    setSettingsBusy(false);
  }
}

async function testModelConnection() {
  setSettingsBusy(true);
  feedback("正在连接模型……");
  const controller = new AbortController();
  try {
    const draft = settingsFromForm();
    const granted = await requestApiPermission(draft.apiUrl);
    if (!granted) {
      throw new Error("未授予模型 API 地址访问权限");
    }
    currentSettings = await saveSettings(draft);
    const client = new OpenAICompatibleClient(currentSettings);
    await client.test(controller.signal);
    feedback("连接成功", "success");
  } catch (error) {
    feedback(error.message, "error");
  } finally {
    setSettingsBusy(false);
  }
}

async function runTask({ task: taskOverride = null, source = "sidepanel" } = {}) {
  if (isTaskBusy()) {
    return;
  }
  const fromControl = source === "control";
  const typedTask = fromControl ? String(taskOverride || "").trim() : elements.taskInput.value.trim();
  const taskImages = fromControl ? [] : pendingImages.map(copyImageAttachment);
  if (!typedTask && taskImages.length === 0) {
    renderError("请输入任务或添加图片");
    if (fromControl) {
      await restoreControlComposer("请输入任务");
    } else {
      elements.taskInput.focus();
    }
    return;
  }
  const task = typedTask || "请根据我附加的图片完成任务。";
  let agentStarted = false;
  let controlNotice = "无法启动任务，请在 Side Panel 查看详细信息。";

  // Lock before the first await. Permission checks, settings persistence and
  // context preparation can all take long enough for another submit event to
  // arrive; activeSession does not exist yet during that window.
  taskRunLocked = true;
  setRunningUi(true, false);
  setStatus(AgentState.THINKING, "正在准备任务");
  await browser.updateControl({
    mode: "starting",
    task,
    think: "正在检查配置…",
  }).catch(() => {});
  try {
    try {
      const draft = settingsFromForm();
      const missing = await validateRequiredConfiguration(draft);
      if (missing.length) {
        throw new Error(missingConfigurationMessage(missing));
      }
      const granted = fromControl
        ? await hasApiPermission(draft.apiUrl)
        : await requestApiPermission(draft.apiUrl);
      if (!granted) {
        throw new Error(fromControl
          ? "请先在 Side Panel 授予模型 API 地址访问权限"
          : "未授予模型 API 地址访问权限");
      }
      currentSettings = await saveSettings(draft);
    } catch (error) {
      controlNotice = error.message;
      elements.settingsPanel.classList.remove("collapsed");
      feedback(error.message, "error");
      setStatus(AgentState.ERROR, "无法启动任务");
      return;
    }

    const modelClient = new OpenAICompatibleClient({
      ...currentSettings,
      onRetry: ({ attempt, delayMs, error }) => {
        const seconds = Math.max(1, Math.ceil(delayMs / 1000));
        const retryLabel = `模型暂时不可用 · ${seconds} 秒后进行第 ${attempt} 次重试`;
        setStatus(
          AgentState.THINKING,
          retryLabel,
        );
        browser.updateControl({ mode: "running", task, think: retryLabel }).catch(() => {});
        console.warn("Venus model request will retry", error);
      },
    });
    const conversationId = currentConversation.id;
    const runId = makeId();
    let conversationContext;
    try {
      conversationContext = await prepareConversationContext(modelClient, async () => {
        setStatus(AgentState.THINKING, "正在压缩上下文");
        const statusMessage = renderMessage("system", "正在压缩上下文…");
        await browser.updateControl({
          mode: "starting",
          task,
          think: "正在压缩上下文…",
        }).catch(() => {});
        return () => statusMessage.remove();
      });
      if (currentConversation.title === "新会话" && currentConversation.nextSequence === 1) {
        currentConversation = await conversationStore.renameConversation(conversationId, task);
        await refreshConversationList();
      }
      await conversationStore.appendEntry(conversationId, {
        kind: "message",
        role: "user",
        text: task,
        attachments: taskImages,
        runId,
      });
    } catch (error) {
      controlNotice = error.message;
      renderError(`无法保存会话：${error.message}`);
      setStatus(AgentState.ERROR, "无法启动任务");
      return;
    }
    renderMessage("user", task, taskImages);
    elements.taskInput.value = "";
    pendingImages = [];
    renderPendingImages();

    activeSession = new AgentSession({
      browser,
      modelClient,
      promptTemplate,
      conversationContext: conversationContext.text,
      conversationImages: conversationContext.images,
      taskImages,
      previousCompletionTokens: currentConversation.completionTokens,
      runId,
      maxSteps: currentSettings.maxSteps,
      onState: ({ state, label }) => {
        setStatus(state, label);
      },
      onAttached: (tab) => {
        elements.tabLabel.textContent = tab.title || tab.url || `Tab ${tab.id}`;
        elements.tabLabel.title = tab.url || "";
        startControlHeartbeat();
      },
      beforeAction: ensureActionPermissions,
      onStep: async (entry) => {
        if (entry.phase === "terminal") stopControlHeartbeat();
        renderStep(entry);
        if (entry.phase === "proposed") {
          await browser.updateControl({
            mode: "running",
            task,
            think: entry.think || "模型未提供分析。",
          });
          await conversationStore.appendEntry(conversationId, stepRecord(entry));
        } else if (["executed", "failed", "terminal"].includes(entry.phase)) {
          await conversationStore.appendEntry(conversationId, {
            kind: "result",
            runId: entry.runId,
            step: entry.step,
            phase: entry.phase,
            result: entry.result,
          });
        }
      },
      onThink: ({ think }) => {
        browser.updateControl({
          mode: "running",
          task,
          think,
        }).catch(() => {});
      },
      onFinal: async ({ type, content, outputTokens, contextTokens }) => {
        const role = type === "call_user" ? "system" : "agent";
        const finalText = content || (type === "call_user" ? "需要用户接管" : "任务完成");
        const text = `${finalText}\n\n${formatTokenStatistics({
          contextTokens,
          outputTokens,
          previousCompletionTokens: activeSession.previousCompletionTokens,
        })}`;
        await conversationStore.appendEntry(conversationId, {
          kind: "message",
          role,
          text,
          runId,
        });
        renderMessage(role, text);
        await browser.updateControl({
          mode: "complete",
          task,
          outcome: type,
          content: finalText,
          noticeTimeoutMs: 3_000,
        });
        setTimeout(() => {
          browser.updateControl({ content: "", noticeTimeoutMs: 0 }).catch(() => {});
        }, 3_000);
      },
    });
    agentStarted = true;
    setRunningUi(true);

    try {
      await activeSession.run(task);
    } catch (error) {
      controlNotice = error.message;
      const errorText = `${error.message}\n\n${formatSessionTokenStatistics(activeSession)}`;
      await conversationStore.appendEntry(conversationId, {
        kind: "message",
        role: "error",
        text: errorText,
        runId,
      }).catch(() => {});
      renderError(errorText);
      await restoreControlComposer(error.message);
    } finally {
      stopControlHeartbeat();
      if (activeSession?.hasPromptTokenUsage) {
        await conversationStore.updatePromptTokens(
          conversationId,
          activeSession.promptTokens,
        ).catch((error) => {
          console.warn("Venus could not persist prompt token usage", error);
        });
      }
      if (activeSession?.hasOutputTokenUsage) {
        await conversationStore.addCompletionTokens(
          conversationId,
          activeSession.outputTokens,
        ).catch((error) => {
          console.warn("Venus could not persist completion token usage", error);
        });
      }
      currentConversation = await conversationStore.getConversation(conversationId).catch(() => currentConversation);
      await refreshConversationList().catch(() => {});
    }
  } finally {
    taskRunLocked = false;
    setRunningUi(false);
    if (!agentStarted) {
      await restoreControlComposer(controlNotice);
    }
  }
}

async function ensureActionPermissions({ action, observation, signal }) {
  if (action?.name !== "download") return;
  await ensureDownloadWorkspace(signal);
  let hostname = "";
  try {
    hostname = new URL(observation?.tab?.url || "").hostname;
  } catch {
    return;
  }
  if (hostname !== "github.com" && hostname !== "raw.githubusercontent.com") return;
  const origins = [
    "https://github.com/*",
    "https://raw.githubusercontent.com/*",
  ];
  if (await chrome.permissions.contains({ origins })) return;
  await waitForPermissionButton(origins, signal);
}

async function ensureDownloadWorkspace(signal) {
  workspaceHandle = workspaceHandle ?? await loadWorkspaceHandle();
  const granted = workspaceHandle
    ? await verifyWorkspacePermission(workspaceHandle, false).catch(() => false)
    : false;
  if (!granted) {
    await waitForWorkspaceButton(signal);
  }
  const workspace = await browser.configureWorkspace();
  if (!workspace.enabled) {
    throw new Error(`文件 Workspace 未启用：${workspace.error}`);
  }
  if (currentConversation) {
    await announceWorkspaceOnce(currentConversation.id, workspace);
  }
}

function waitForWorkspaceButton(signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new Error("任务已停止"));
      return;
    }
    elements.emptyState.classList.add("hidden");
    const message = document.createElement("div");
    message.className = "message system permission-request";
    const text = document.createElement("div");
    text.className = "message-text";
    text.textContent = workspaceHandle
      ? "下载文件需要重新授权 Workspace 目录，授权后任务会自动继续。"
      : "下载文件需要设置 Workspace 目录，选择后任务会自动继续。";
    const button = document.createElement("button");
    button.className = "primary-button permission-button";
    button.type = "button";
    button.textContent = workspaceHandle ? "授权并继续" : "选择目录并继续";
    message.append(text, button);
    elements.messages.append(message);
    scrollConversationToBottom();

    const cleanup = () => signal?.removeEventListener("abort", onAbort);
    const onAbort = () => {
      cleanup();
      button.disabled = true;
      text.textContent = "Workspace 设置已取消。";
      reject(signal.reason ?? new Error("任务已停止"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    button.addEventListener("click", async () => {
      button.disabled = true;
      button.textContent = "正在设置……";
      try {
        await chooseWorkspace();
        cleanup();
        text.textContent = `Workspace 已就绪：${workspaceHandle.name}，正在继续下载。`;
        button.remove();
        resolve();
      } catch (error) {
        text.textContent = `Workspace 设置失败：${error.message}`;
        button.textContent = workspaceHandle ? "重新授权" : "重新选择";
        button.disabled = false;
      }
    });
  });
}

function waitForPermissionButton(origins, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new Error("任务已停止"));
      return;
    }
    elements.emptyState.classList.add("hidden");
    const message = document.createElement("div");
    message.className = "message system permission-request";
    const text = document.createElement("div");
    text.className = "message-text";
    text.textContent = "即将从 GitHub 下载文件。请授权本次下载源访问权限，授权后任务会自动继续。";
    const button = document.createElement("button");
    button.className = "primary-button permission-button";
    button.type = "button";
    button.textContent = "授予权限并继续";
    message.append(text, button);
    elements.messages.append(message);
    scrollConversationToBottom();

    const cleanup = () => signal?.removeEventListener("abort", onAbort);
    const onAbort = () => {
      cleanup();
      button.disabled = true;
      text.textContent = "权限申请已取消。";
      reject(signal.reason ?? new Error("任务已停止"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    button.addEventListener("click", async () => {
      button.disabled = true;
      button.textContent = "正在请求……";
      try {
        const granted = await chrome.permissions.request({ origins });
        if (!granted) throw new Error("用户未授予 GitHub 下载权限");
        cleanup();
        text.textContent = "GitHub 下载权限已授予，任务正在继续。";
        button.remove();
        resolve();
      } catch (error) {
        text.textContent = `权限申请失败：${error.message}`;
        button.textContent = "重新授权";
        button.disabled = false;
      }
    });
  });
}

async function restoreControlComposer(notice = "", noticeTimeoutMs = 0) {
  await browser.updateControl({
    mode: "ready",
    task: "Venus Browser Agent",
    think: "输入新任务",
    notice,
    noticeTimeoutMs,
  }).catch(() => {});
}

async function stopTask(message = "用户已停止任务，浏览器控制已释放。") {
  elements.stopTask.disabled = true;
  stopControlHeartbeat();
  try {
    await activeSession?.stop();
    const text = `${String(message || "用户已停止任务，浏览器控制已释放。")}\n\n${formatSessionTokenStatistics(activeSession)}`;
    if (currentConversation) {
      await conversationStore.appendEntry(currentConversation.id, {
        kind: "message",
        role: "system",
        text,
      });
    }
    renderMessage("system", text);
    setStatus(AgentState.STOPPED, "已停止");
    await restoreControlComposer(text, 3_000);
    setTimeout(() => {
      browser.updateControl({ notice: "", noticeTimeoutMs: 0 }).catch(() => {});
    }, 3_000);
  } finally {
    setRunningUi(false);
    elements.stopTask.disabled = false;
  }
}

function formatSessionTokenStatistics(session) {
  return formatTokenStatistics({
    contextTokens: session?.hasPromptTokenUsage ? session.contextTokens : null,
    outputTokens: session?.hasOutputTokenUsage ? session.outputTokens : null,
    previousCompletionTokens: session?.previousCompletionTokens,
  });
}

function formatTokenStatistics({
  contextTokens = null,
  outputTokens = null,
  previousCompletionTokens = 0,
} = {}) {
  const contextText = contextTokens == null
    ? "当前上下文 token：未提供"
    : `当前上下文 token：${Number(contextTokens).toLocaleString("en-US")}`;
  const previous = Math.max(0, Number(previousCompletionTokens) || 0);
  const completionText = outputTokens == null
    ? (previous > 0
        ? `会话累计 completion token：${previous.toLocaleString("en-US")}（本任务未提供，统计可能不完整）`
        : "会话累计 completion token：未提供")
    : `会话累计 completion token：${(previous + Number(outputTokens)).toLocaleString("en-US")}`;
  return `${contextText}\n${completionText}`;
}

function startControlHeartbeat() {
  stopControlHeartbeat();
  const renew = () => browser.heartbeat().catch(() => {});
  renew();
  controlHeartbeatTimer = setInterval(renew, 2_000);
}

function stopControlHeartbeat() {
  if (controlHeartbeatTimer === null) return;
  clearInterval(controlHeartbeatTimer);
  controlHeartbeatTimer = null;
}

function renderStep(entry) {
  const stepKey = `${entry.runId ?? "legacy"}:${entry.step}`;
  let card = stepCards.get(stepKey);
  if (!card) {
    card = document.createElement("article");
    card.className = "step-card";

    const header = document.createElement("div");
    header.className = "step-header";
    const number = document.createElement("span");
    number.className = "step-number";
    number.textContent = `STEP ${entry.step}`;
    const page = document.createElement("span");
    page.className = "step-page";
    page.textContent = compactHost(entry.tab?.url);
    const phase = document.createElement("span");
    phase.className = "step-phase";
    phase.textContent = "待执行";
    header.append(number, page, phase);

    const action = document.createElement("div");
    action.className = "step-action";
    action.textContent = entry.rawAction;

    const think = document.createElement("p");
    think.className = "step-think";
    think.textContent = entry.think || "模型未提供分析。";

    const result = document.createElement("pre");
    result.className = "step-result hidden";
    card.append(header, action, think, result);
    card.refs = { phase, result };
    stepCards.set(stepKey, card);
    elements.messages.append(card);
  }

  if (["executed", "failed", "terminal"].includes(entry.phase)) {
    card.refs.phase.textContent = entry.phase === "terminal"
      ? "已完成"
      : entry.phase === "failed"
        ? "执行失败"
        : "已执行";
    card.refs.result.classList.remove("hidden");
    card.refs.result.textContent = compactJson(entry.result);
  }
  scrollConversationToBottom();
}

function renderMessage(role, text, attachments = []) {
  elements.emptyState.classList.add("hidden");
  const message = document.createElement("div");
  message.className = `message ${role}`;
  const messageText = document.createElement("div");
  messageText.className = "message-text";
  messageText.textContent = String(text ?? "");
  message.append(messageText);
  const images = normalizeStoredImages(attachments);
  if (images.length) {
    const gallery = document.createElement("div");
    gallery.className = "message-images";
    for (const attachment of images) {
      const item = document.createElement("div");
      const image = document.createElement("img");
      image.className = "message-image";
      image.src = attachment.dataUrl;
      image.alt = attachment.name || "用户图片";
      const name = document.createElement("span");
      name.className = "message-image-name";
      name.textContent = attachment.name || "图片";
      name.title = attachment.name || "图片";
      item.append(image, name);
      gallery.append(item);
    }
    message.append(gallery);
  }
  elements.messages.append(message);
  scrollConversationToBottom();
  return message;
}

async function addImageFiles(fileList) {
  const files = [...(fileList ?? [])];
  if (!files.length) return;
  if (pendingImages.length + files.length > MAX_IMAGES) {
    throw new Error(`每条消息最多添加 ${MAX_IMAGES} 张图片`);
  }

  const typedFiles = files.map((file) => ({ file, mimeType: imageMimeType(file) }));
  for (const { file, mimeType } of typedFiles) {
    if (!IMAGE_TYPES.has(mimeType)) {
      throw new Error(`不支持的图片格式：${file.name || file.type || "unknown"}`);
    }
    if (file.size > MAX_IMAGE_BYTES) {
      throw new Error(`图片 ${file.name || "未命名图片"} 超过 10 MiB`);
    }
  }
  const totalBytes = pendingImages.reduce((total, image) => total + image.size, 0)
    + typedFiles.reduce((total, item) => total + item.file.size, 0);
  if (totalBytes > MAX_IMAGE_TOTAL_BYTES) {
    throw new Error("待发送图片总大小不能超过 20 MiB");
  }

  const attachments = await Promise.all(typedFiles.map(({ file, mimeType }) => readImageFile(file, mimeType)));
  pendingImages.push(...attachments);
  renderPendingImages();
}

async function readImageFile(file, mimeType) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunkSize = 32 * 1024;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return {
    id: makeId(),
    name: String(file.name || `image-${Date.now()}.${imageExtension(mimeType)}`),
    mimeType,
    size: file.size,
    dataUrl: `data:${mimeType};base64,${btoa(binary)}`,
  };
}

function renderPendingImages() {
  elements.imagePreview.replaceChildren();
  elements.imagePreview.classList.toggle("hidden", pendingImages.length === 0);
  for (const attachment of pendingImages) {
    const item = document.createElement("div");
    item.className = "image-preview-item";
    const image = document.createElement("img");
    image.src = attachment.dataUrl;
    image.alt = attachment.name;
    const name = document.createElement("span");
    name.textContent = attachment.name;
    name.title = attachment.name;
    const remove = document.createElement("button");
    remove.className = "remove-image";
    remove.type = "button";
    remove.textContent = "×";
    remove.title = `移除 ${attachment.name}`;
    remove.disabled = isTaskBusy();
    remove.addEventListener("click", () => {
      pendingImages = pendingImages.filter((imageAttachment) => imageAttachment.id !== attachment.id);
      renderPendingImages();
    });
    item.append(image, remove, name);
    elements.imagePreview.append(item);
  }
}

function clearPendingImages() {
  pendingImages = [];
  elements.imageInput.value = "";
  renderPendingImages();
}

function imageMimeType(file) {
  const declared = String(file?.type || "").toLowerCase();
  if (IMAGE_TYPES.has(declared)) return declared;
  const extension = String(file?.name || "").split(".").at(-1)?.toLowerCase();
  return ({ jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp", gif: "image/gif" })[extension] || declared;
}

function imageExtension(mimeType) {
  return mimeType === "image/jpeg" ? "jpg" : mimeType.split("/").at(-1) || "png";
}

function copyImageAttachment(attachment) {
  return {
    id: String(attachment.id || makeId()),
    name: String(attachment.name || "图片"),
    mimeType: String(attachment.mimeType || "image/png"),
    size: Number(attachment.size) || 0,
    dataUrl: String(attachment.dataUrl || ""),
  };
}

function normalizeStoredImages(attachments) {
  return (Array.isArray(attachments) ? attachments : [])
    .filter((attachment) => IMAGE_TYPES.has(String(attachment?.mimeType || "").toLowerCase()))
    .filter((attachment) => /^data:image\/(?:png|jpeg|webp|gif);base64,/i.test(String(attachment?.dataUrl || "")))
    .map(copyImageAttachment);
}

function recentConversationImages(entries) {
  const images = [];
  let totalBytes = 0;
  for (let index = entries.length - 1; index >= 0 && images.length < MAX_IMAGES; index -= 1) {
    const attachments = normalizeStoredImages(entries[index]?.attachments).reverse();
    for (const attachment of attachments) {
      if (images.length >= MAX_IMAGES) break;
      if (totalBytes + attachment.size > MAX_IMAGE_TOTAL_BYTES) continue;
      images.unshift(attachment);
      totalBytes += attachment.size;
    }
  }
  return images;
}

function renderError(text) {
  renderMessage("error", text);
}

function renderFileTransferEvent(event) {
  if (event.type === "upload" && event.status === "selected") {
    renderMessage("system", `已从 workspace 选择上传文件：${event.files.join(", ")}`);
  } else if (event.type === "upload" && event.status === "failed") {
    renderError(`自动选择上传文件失败：${event.error}`);
  } else if (event.type === "download" && event.status === "completed") {
    renderMessage("system", `下载完成：${event.filename} → ${event.workspace}`);
  } else if (event.type === "download" && event.status === "canceled") {
    renderError(`下载已取消：${event.filename}`);
  }
}

function clearRenderedConversation() {
  elements.messages.replaceChildren();
  stepCards.clear();
  elements.emptyState.classList.remove("hidden");
}

async function startNewConversation() {
  if (isTaskBusy()) return;
  clearPendingImages();
  currentConversation = await conversationStore.createConversation();
  await refreshConversationList();
  clearRenderedConversation();
  setStatus(AgentState.IDLE, "新会话");
  elements.taskInput.focus();
}

async function switchConversation() {
  if (isTaskBusy()) return;
  clearPendingImages();
  currentConversation = await conversationStore.setActiveConversation(
    elements.conversationSelect.value,
  );
  await restoreConversation(currentConversation.id);
  setStatus(AgentState.IDLE, "已恢复会话");
}

async function deleteCurrentConversation() {
  if (isTaskBusy() || !currentConversation) return;
  if (!window.confirm(`删除会话“${currentConversation.title}”？此操作无法撤销。`)) return;
  workspaceNoticeConversations.delete(currentConversation.id);
  await conversationStore.deleteConversation(currentConversation.id);
  clearPendingImages();
  currentConversation = await conversationStore.getOrCreateActiveConversation();
  await refreshConversationList();
  await restoreConversation(currentConversation.id);
  setStatus(AgentState.IDLE, "会话已删除");
}

async function refreshConversationList() {
  const conversations = await conversationStore.listConversations();
  elements.conversationSelect.replaceChildren();
  for (const conversation of conversations) {
    const option = document.createElement("option");
    option.value = conversation.id;
    option.textContent = conversation.title;
    option.title = `${conversation.title} · ${formatDate(conversation.updatedAt)}`;
    option.selected = conversation.id === currentConversation?.id;
    elements.conversationSelect.append(option);
  }
}

async function restoreConversation(conversationId) {
  clearRenderedConversation();
  const entries = await conversationStore.listEntries(conversationId);
  const hasWorkspaceNotice = entries.some((entry) => (
    entry.kind === "message"
    && entry.role === "system"
    && (entry.messageType === "workspace_ready" || String(entry.text || "").startsWith("文件 workspace 已就绪："))
  ));
  if (hasWorkspaceNotice) workspaceNoticeConversations.add(conversationId);
  else workspaceNoticeConversations.delete(conversationId);
  const visibleEntries = entries.slice(-300);
  if (visibleEntries.length < entries.length) {
    renderMessage("system", `较早的 ${entries.length - visibleEntries.length} 条记录已折叠，但仍保存在本地 transcript 中。`);
  }
  for (const entry of visibleEntries) {
    if (entry.kind === "message") {
      renderMessage(entry.role, entry.text, entry.attachments);
    } else if (entry.kind === "step") {
      renderStep({ ...entry, phase: "proposed" });
    } else if (entry.kind === "result") {
      renderStep({ ...entry, phase: entry.phase ?? "executed" });
    }
  }
  if (entries.length === 0) {
    elements.emptyState.classList.remove("hidden");
  }
}

async function prepareConversationContext(modelClient, onCompacting = async () => {}) {
  currentConversation = await conversationStore.getConversation(currentConversation.id);
  let entries = await conversationStore.listEntries(currentConversation.id, {
    afterSequence: currentConversation.summaryThrough,
  });
  const plan = planCompaction(currentConversation, entries);
  if (plan) {
    let finishCompacting = () => {};
    try {
      finishCompacting = await onCompacting() || finishCompacting;
      const summary = await modelClient.summarize(
        buildCompactionMessages(currentConversation, plan),
      );
      currentConversation = await conversationStore.updateSummary(
        currentConversation.id,
        summary,
        plan.throughSequence,
      );
      entries = entries.filter((entry) => entry.sequence > plan.throughSequence);
    } catch (error) {
      console.warn("Venus context compaction failed; using bounded recent history", error);
    } finally {
      await finishCompacting();
    }
  }
  return {
    text: buildConversationContext(currentConversation, entries),
    images: recentConversationImages(entries),
  };
}

function stepRecord(entry) {
  return {
    kind: "step",
    runId: entry.runId,
    step: entry.step,
    tab: entry.tab,
    viewport: entry.viewport,
    think: entry.think,
    rawResponse: entry.rawResponse,
    rawAction: entry.rawAction,
    action: entry.action,
  };
}

function settingsFromForm() {
  return {
    apiUrl: elements.apiUrl.value,
    model: elements.model.value,
    maxSteps: elements.maxSteps.value,
    temperature: elements.temperature.value,
    apiKey: elements.apiKey.value,
    rememberKey: elements.rememberKey.checked,
  };
}

async function validateRequiredConfiguration(settings) {
  return missingConfigurationFields(settings);
}

async function announceWorkspaceOnce(conversationId, workspace) {
  if (workspaceNoticeConversations.has(conversationId)) return;
  workspaceNoticeConversations.add(conversationId);
  const text = `文件 workspace 已就绪：${workspace.name}`;
  await conversationStore.appendEntry(conversationId, {
    kind: "message",
    role: "system",
    messageType: "workspace_ready",
    text,
  }).catch((error) => {
    console.warn("Venus could not persist the workspace-ready notice", error);
  });
  renderMessage("system", text);
}

function populateSettings(settings) {
  elements.apiUrl.value = settings.apiUrl ?? "";
  elements.model.value = settings.model ?? "";
  elements.maxSteps.value = String(settings.maxSteps ?? 100);
  elements.temperature.value = String(settings.temperature ?? 0);
  elements.apiKey.value = settings.apiKey ?? "";
  elements.rememberKey.checked = Boolean(settings.rememberKey);
  elements.apiUrl.disabled = false;
  elements.model.disabled = false;
  elements.apiKey.disabled = false;
  elements.rememberKey.disabled = false;
}

async function chooseWorkspace() {
  if (typeof window.showDirectoryPicker !== "function") {
    throw new Error("当前 Chrome 不支持目录授权 API");
  }

  // A persisted handle in the `prompt` state only needs its permission
  // restored; making the user find and select the same directory again is
  // unnecessary.
  if (workspaceHandle) {
    const permission = await workspaceHandle.queryPermission({ mode: "readwrite" });
    if (permission === "prompt") {
      if (!await verifyWorkspacePermission(workspaceHandle, true)) {
        throw new Error("未授予 workspace 读写权限");
      }
      await refreshWorkspaceUi();
      feedback(`Workspace 授权已恢复：${workspaceHandle.name}`, "success");
      return;
    }
  }

  const handle = await window.showDirectoryPicker({
    id: "venus-workspace",
    mode: "readwrite",
  });
  if (!await verifyWorkspacePermission(handle, true)) {
    throw new Error("未授予 workspace 读写权限");
  }
  await saveWorkspaceHandle(handle);
  workspaceHandle = handle;
  await refreshWorkspaceUi();
  feedback(`Workspace 已设置为：${handle.name}`, "success");
}

async function refreshWorkspaceUi() {
  const handle = await loadWorkspaceHandle();
  workspaceHandle = handle;
  if (!handle) {
    elements.workspaceLabel.textContent = "尚未选择目录";
    elements.chooseWorkspace.textContent = "选择目录";
    return;
  }
  const granted = await verifyWorkspacePermission(handle, false);
  elements.workspaceLabel.textContent = granted ? `${handle.name} · 已授权` : `${handle.name} · 需要确认访问权限`;
  elements.workspaceLabel.title = handle.name;
  elements.chooseWorkspace.textContent = granted ? "更改目录" : "确认授权";
}

function setSettingsBusy(busy) {
  elements.saveSettings.disabled = busy;
  elements.testModel.disabled = busy;
}

function feedback(text, kind = "") {
  elements.settingsFeedback.textContent = text;
  elements.settingsFeedback.className = `feedback ${kind}`.trim();
}

function isTaskBusy() {
  return taskRunLocked || Boolean(activeSession?.running);
}

function setRunningUi(running, stoppable = running) {
  elements.runTask.disabled = running;
  elements.stopTask.classList.toggle("hidden", !stoppable);
  elements.taskInput.disabled = running;
  elements.clearChat.disabled = running;
  elements.conversationSelect.disabled = running;
  elements.deleteConversation.disabled = running;
  elements.chooseWorkspace.disabled = running;
  elements.addImage.disabled = running;
  elements.imageInput.disabled = running;
  for (const button of elements.imagePreview.querySelectorAll(".remove-image")) {
    button.disabled = running;
  }
}

function setStatus(state, label) {
  const runningStates = [AgentState.ATTACHING, AgentState.OBSERVING, AgentState.THINKING, AgentState.EXECUTING];
  const className = runningStates.includes(state)
    ? "running"
    : state === AgentState.FINISHED
      ? "success"
      : state === AgentState.WAITING_USER
        ? "warning"
        : state === AgentState.ERROR
          ? "error"
          : "idle";
  elements.statusDot.className = `status-dot ${className}`;
  elements.statusLabel.textContent = label || state;
}

function compactJson(value) {
  const text = JSON.stringify(value, null, 2);
  return text.length <= 1400 ? text : `${text.slice(0, 1400)}\n…`;
}

function compactHost(rawUrl) {
  try {
    return new URL(rawUrl).hostname;
  } catch {
    return "page";
  }
}

function scrollConversationToBottom() {
  requestAnimationFrame(() => {
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
  });
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "" : date.toLocaleString();
}

function makeId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function handleUiError(error) {
  renderError(error?.message ?? String(error));
  setStatus(AgentState.ERROR, "操作失败");
}

function byId(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element #${id}`);
  }
  return element;
}
