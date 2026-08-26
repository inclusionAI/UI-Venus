/**
 * Usage: Open the Side Panel, enter a task in Page Assistant, and click the left icon during execution to stop.
 */
import {
  findWorkspaceFile,
  githubFileDownloadTarget,
  isLikelyDownloadLink,
  listAcceptedWorkspaceFiles,
  normalizeDownloadFilename,
  normalizeDownloadResourceUrl,
  stabilizeDownloadUrl,
} from "./src/file-transfer.js";
import {
  createWorkspaceFile,
  getWorkspaceFile,
  listWorkspaceFiles,
  loadWorkspaceHandle,
  verifyWorkspacePermission,
} from "./src/workspace-store.js";
import { editingCommandsForHotkey, normalizeHotkeyKeys } from "./src/hotkey.js";

const sessions = new Map();
const pageAssistantStates = new Map();
const pendingPageTasks = new Map();
const hiddenPageAssistantTabs = new Set();
const sidePanelCloseTimers = new Map();
let sessionSequence = 0;

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
  await chrome.storage.local.setAccessLevel({ accessLevel: "TRUSTED_CONTEXTS" });
  await chrome.storage.session.setAccessLevel({ accessLevel: "TRUSTED_CONTEXTS" });
});

chrome.sidePanel.onOpened?.addListener((info) => {
  clearTimeout(sidePanelCloseTimers.get(info.windowId));
  sidePanelCloseTimers.delete(info.windowId);
  handleNativeSidePanelVisibility(info, true).catch(() => {});
});

chrome.sidePanel.onClosed?.addListener((info) => {
  clearTimeout(sidePanelCloseTimers.get(info.windowId));
  const timer = setTimeout(async () => {
    sidePanelCloseTimers.delete(info.windowId);
    if (Number.isInteger(info.tabId)) {
      const [activeTab] = await chrome.tabs.query({ active: true, windowId: info.windowId });
      if (activeTab?.id !== info.tabId) return;
    }
    await handleNativeSidePanelVisibility(info, false);
  }, 250);
  sidePanelCloseTimers.set(info.windowId, timer);
});

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "venus-agent-session") {
    return;
  }

  const sequence = ++sessionSequence;
  const session = {
    id: `venus-${Date.now()}-${sequence}`,
    port,
    attachedTabs: new Set(),
    ownedTabs: new Set(),
    activeTabId: null,
    bootstrapTabId: null,
    windowId: null,
    viewport: null,
    selectContexts: new Map(),
    workspace: disabledWorkspace(),
    clickInteraction: null,
    pendingFileChoosers: new Map(),
    downloads: new Map(),
    pendingTransferNotices: [],
    controlIndicators: new Map(),
    stopRequested: false,
    taskLabel: "",
    detachPromise: null,
    controlState: readyControlState(),
    controlEnabled: true,
    panelVisible: false,
    assistantTabId: null,
    closed: false,
  };
  sessions.set(session.id, session);

  port.onMessage.addListener((message) => {
    handleRequest(session, message).catch((error) => {
      respond(session, message?.requestId, false, null, humanizeError(error));
    });
  });

  port.onDisconnect.addListener(() => {
    if (["running", "starting"].includes(session.controlState.mode) && session.assistantTabId) {
      sendPageAssistantUpdate(session.assistantTabId, {
        mode: "ready",
        task: "Venus Browser Agent",
        think: "输入新任务",
        notice: "Side Panel 已关闭，当前任务已停止。",
      });
    }
    if (session.assistantTabId) {
      hiddenPageAssistantTabs.add(session.assistantTabId);
      setPageAssistantEnabled(session.assistantTabId, false);
    }
    session.closed = true;
    sessions.delete(session.id);
    detachAll(session);
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const tabId = sender.tab?.id;
  if (!Number.isInteger(tabId)) return false;
  if (message?.type === "venus_page_assistant_ready") {
    chrome.storage.local.get(["controlPanelEnabled"]).then((stored) => {
      const visibleSession = [...sessions.values()].reverse().find((session) => {
        if (session.closed || !session.panelVisible || !session.controlEnabled) return false;
        if (session.windowId !== sender.tab.windowId) return false;
        const busy = ["running", "starting"].includes(session.controlState.mode);
        return !busy || session.assistantTabId === tabId;
      });
      sendResponse({
        enabled: Boolean(visibleSession)
          && stored.controlPanelEnabled !== false
          && !hiddenPageAssistantTabs.has(tabId),
        state: visibleSession?.controlState
          ?? pageAssistantStates.get(tabId)
          ?? readyControlState(),
      });
    });
    return true;
  }
  if (message?.type === "venus_page_assistant_event") {
    handlePageAssistantEvent(tabId, message).catch((error) => {
      sendPageAssistantUpdate(tabId, {
        mode: "ready",
        notice: humanizeError(error),
      });
    });
  }
  return false;
});

chrome.debugger.onDetach.addListener((source, reason) => {
  const tabId = source.tabId;
  if (!Number.isInteger(tabId)) {
    return;
  }
  for (const session of sessions.values()) {
    if (!session.attachedTabs.delete(tabId)) {
      continue;
    }
    session.selectContexts.delete(tabId);
    if (session.activeTabId === tabId) {
      session.activeTabId = null;
    }
    sendEvent(session, "debugger_detached", {
      tabId,
      reason,
      recoverable: reason === "target_closed",
    });
  }
});

chrome.debugger.onEvent.addListener((source, method, params) => {
  handleDebuggerEvent(source, method, params).catch((error) => {
    console.warn("Venus browser event failed", error);
  });
});

chrome.tabs.onRemoved.addListener((tabId) => {
  pageAssistantStates.delete(tabId);
  pendingPageTasks.delete(tabId);
  hiddenPageAssistantTabs.delete(tabId);
  for (const session of sessions.values()) {
    session.attachedTabs.delete(tabId);
    session.selectContexts.delete(tabId);
    if (session.activeTabId === tabId) {
      session.activeTabId = null;
    }
  }
});

chrome.tabs.onActivated.addListener(({ tabId, windowId }) => {
  syncIdlePageAssistantTab(tabId, windowId).catch(() => {});
});

chrome.downloads.onCreated.addListener((item) => {
  handleNativeDownloadCreated(item).catch((error) => {
    console.warn("Venus could not cancel a native download", error);
  });
});

async function handleRequest(session, message) {
  if (!message?.requestId || !message?.type) {
    return;
  }

  let result;
  switch (message.type) {
    case "configure_workspace":
      result = await configureWorkspace(session);
      break;
    case "show_control":
      result = await showIdleControl(session);
      break;
    case "claim_page_task":
      result = await claimPendingPageTask(session, message.tabId);
      break;
    case "set_panel_visibility":
      result = await setPanelVisibility(session, message.visible);
      break;
    case "set_control_enabled":
      result = await setControlEnabled(session, message.enabled);
      break;
    case "attach_current":
      result = await attachCurrentTab(session, message.task);
      break;
    case "capture":
      result = await captureObservation(session);
      break;
    case "execute":
      result = await executeAction(session, message.action);
      break;
    case "update_control":
      result = await updateControlIndicator(session, message.payload);
      break;
    case "detach":
      await detachAll(session);
      result = { detached: true };
      break;
    case "control_heartbeat":
      await renewControlIndicatorLeases(session);
      result = { renewed: true };
      break;
    default:
      throw new Error(`未知浏览器命令：${message.type}`);
  }
  respond(session, message.requestId, true, result);
}

function readyControlState() {
  return {
    mode: "ready",
    task: "Venus Browser Agent",
    think: "输入新任务",
    outcome: "finished",
    content: "",
    notice: "",
    noticeTimeoutMs: 0,
  };
}

async function sendPageAssistantUpdate(tabId, payload) {
  const state = normalizeControlState(payload, pageAssistantStates.get(tabId) ?? readyControlState());
  pageAssistantStates.set(tabId, state);
  const message = { type: "venus_page_assistant_update", state };
  try {
    await chrome.tabs.sendMessage(tabId, message);
  } catch {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["page-assistant.js"],
    }).catch(() => {});
    await chrome.tabs.sendMessage(tabId, message).catch(() => {});
  }
  return state;
}

async function handlePageAssistantEvent(tabId, message) {
  const openSessions = [...sessions.values()].reverse();
  const matchingSession = openSessions.find((candidate) => (
    candidate.assistantTabId === tabId || candidate.activeTabId === tabId
  ));
  if (message.event === "stop") {
    if (matchingSession) sendEvent(matchingSession, "stop_requested", { tabId });
    return;
  }
  if (message.event !== "run_task") return;
  const task = String(message.task || "").trim().slice(0, 10_000);
  if (!task) return;
  const session = matchingSession ?? openSessions[0];
  pendingPageTasks.set(tabId, task);
  if (!session || session.closed) {
    // Chrome only preserves the click's user gesture until the first async
    // boundary. Invoke sidePanel.open synchronously, before configuration IO.
    const openRequest = chrome.sidePanel.open({ tabId });
    try {
      await openRequest;
    } catch (error) {
      // The panel can already be visible while its runtime port is still
      // connecting. Give that connection a chance to consume the queued task.
      await sleep(500);
      if (!pendingPageTasks.has(tabId)) return;
      pendingPageTasks.delete(tabId);
      throw error;
    }
    return;
  }
  if (["running", "starting"].includes(session.controlState.mode)) {
    pendingPageTasks.delete(tabId);
    await sendPageAssistantUpdate(tabId, {
      mode: "ready",
      task: "Venus Browser Agent",
      think: "输入新任务",
      notice: "已有任务正在运行，请等待当前任务结束或先停止任务。",
    });
    return;
  }
  session.assistantTabId = tabId;
  sendEvent(session, "page_task_available", { tabId });
}

function takePendingPageTask(session, tab) {
  const task = Number.isInteger(tab?.id) ? pendingPageTasks.get(tab.id) : null;
  if (!task) return null;
  pendingPageTasks.delete(tab.id);
  session.assistantTabId = tab.id;
  session.controlState = normalizeControlState({
    mode: "starting",
    task,
    think: "正在启动…",
  }, readyControlState());
  return { tabId: tab.id, task };
}

async function handleNativeSidePanelVisibility(info, visible) {
  const matchingSessions = [...sessions.values()].filter((session) => {
    if (Number.isInteger(info?.windowId)) return session.windowId === info.windowId;
    return Number.isInteger(info?.tabId) && session.assistantTabId === info.tabId;
  });
  await Promise.allSettled(matchingSessions.map((session) => (
    setPanelVisibility(session, visible)
  )));
}

async function showIdleControl(session) {
  session.controlEnabled = true;
  session.panelVisible = true;
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!tab?.id || !isControllableUrl(tab.url)) {
    return { shown: false };
  }
  session.assistantTabId = tab.id;
  session.windowId = tab.windowId;
  hiddenPageAssistantTabs.delete(tab.id);
  session.controlState = readyControlState();
  await sendPageAssistantUpdate(tab.id, session.controlState);
  return { shown: true, tabId: tab.id };
}

async function claimPendingPageTask(session, requestedTabId = null) {
  const requested = Number(requestedTabId);
  const queuedTabId = Number.isInteger(requested) && pendingPageTasks.has(requested)
    ? requested
    : [...pendingPageTasks.keys()].at(-1);
  if (!Number.isInteger(queuedTabId)) return { pendingTask: null };
  const tab = await chrome.tabs.get(queuedTabId).catch(() => null);
  if (!tab) {
    pendingPageTasks.delete(queuedTabId);
    return { pendingTask: null };
  }
  const pendingTask = takePendingPageTask(session, tab);
  if (pendingTask) {
    session.panelVisible = true;
    session.windowId = tab.windowId;
  }
  return { pendingTask };
}

async function setPanelVisibility(session, visible) {
  session.panelVisible = Boolean(visible);
  if (!session.assistantTabId) return { visible: session.panelVisible };
  const shouldShow = session.panelVisible && session.controlEnabled;
  if (shouldShow) hiddenPageAssistantTabs.delete(session.assistantTabId);
  else hiddenPageAssistantTabs.add(session.assistantTabId);
  await setPageAssistantEnabled(session.assistantTabId, shouldShow);
  if (shouldShow) {
    await sendPageAssistantUpdate(session.assistantTabId, session.controlState);
  }
  return { visible: session.panelVisible };
}

async function setControlEnabled(session, enabled) {
  session.controlEnabled = Boolean(enabled);
  if (session.controlEnabled && session.assistantTabId) {
    await sendPageAssistantUpdate(session.assistantTabId, session.controlState);
  }
  if (session.assistantTabId) {
    if (session.controlEnabled) hiddenPageAssistantTabs.delete(session.assistantTabId);
    else hiddenPageAssistantTabs.add(session.assistantTabId);
    await setPageAssistantEnabled(
      session.assistantTabId,
      session.controlEnabled && session.panelVisible,
    );
  }
  return { enabled: session.controlEnabled };
}

async function attachCurrentTab(session, task = "") {
  await detachAll(session);
  session.stopRequested = false;
  session.taskLabel = String(task || "").trim();
  session.controlState = {
    mode: "running",
    task: session.taskLabel,
    think: "正在思考…",
  };
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!tab?.id) {
    throw new Error("没有可接管的当前标签页");
  }
  await bindPageAssistantToTab(session, tab);
  if (!isControllableUrl(tab.url)) {
    if (!isBootstrapUrl(tab.url || tab.pendingUrl)) {
      assertControllableUrl(tab.url);
    }
    session.bootstrapTabId = tab.id;
    session.windowId = tab.windowId;
    return tabSummary(tab);
  }
  await attachTab(session, tab.id);
  session.activeTabId = tab.id;
  session.windowId = tab.windowId;
  return tabSummary(tab);
}

async function bindPageAssistantToTab(session, tab) {
  if (!Number.isInteger(tab?.id)) return;
  const previousTabId = session.assistantTabId;
  if (Number.isInteger(previousTabId) && previousTabId !== tab.id) {
    hiddenPageAssistantTabs.add(previousTabId);
    await setPageAssistantEnabled(previousTabId, false);
  }
  session.assistantTabId = tab.id;
  session.windowId = tab.windowId;
  if (session.attachedTabs.has(tab.id)) {
    await sendCdp(tab.id, "Runtime.evaluate", {
      expression: `document.getElementById('__venus_control_indicator__')?.remove()`,
    }).catch(() => {});
  }
  const shouldShow = session.controlEnabled && session.panelVisible;
  if (shouldShow) {
    hiddenPageAssistantTabs.delete(tab.id);
    await sendPageAssistantUpdate(tab.id, session.controlState);
  } else {
    hiddenPageAssistantTabs.add(tab.id);
    await setPageAssistantEnabled(tab.id, false);
  }
}

async function syncIdlePageAssistantTab(tabId, windowId) {
  const tab = await chrome.tabs.get(tabId).catch(() => null);
  const matchingSessions = [...sessions.values()].filter((session) => (
    !session.closed
    && session.windowId === windowId
    && session.panelVisible
    && session.controlEnabled
    && !["running", "starting"].includes(session.controlState.mode)
  ));
  for (const session of matchingSessions) {
    if (tab && isControllableUrl(tab.url)) {
      await bindPageAssistantToTab(session, tab);
      continue;
    }
    if (Number.isInteger(session.assistantTabId)) {
      hiddenPageAssistantTabs.add(session.assistantTabId);
      await setPageAssistantEnabled(session.assistantTabId, false);
    }
    session.assistantTabId = null;
  }
}

async function attachTab(session, tabId, { controlOnly = false } = {}) {
  if (session.attachedTabs.has(tabId)) {
    return;
  }
  try {
    await chrome.debugger.attach({ tabId }, "1.3");
  } catch (error) {
    throw new Error(`无法接管标签页。请关闭该页面的 DevTools 后重试：${error.message}`);
  }
  session.attachedTabs.add(tabId);
  session.ownedTabs.add(tabId);
  try {
    await sendCdp(tabId, "Page.enable");
    await sendCdp(tabId, "Runtime.enable");
    if (!controlOnly) {
      await configureFileTransferForTab(session, tabId);
    }
  } catch (error) {
    session.attachedTabs.delete(tabId);
    await safeDetach(tabId);
    throw error;
  }
}

async function installControlIndicator(session, tabId) {
  const source = controlIndicatorSource(
    session.controlState,
    session.controlBinding,
    session.controlToken,
  );
  try {
    const installed = await sendCdp(tabId, "Page.addScriptToEvaluateOnNewDocument", { source });
    if (installed.identifier) {
      session.controlIndicators.set(tabId, installed.identifier);
    }
    await sendCdp(tabId, "Runtime.evaluate", { expression: source });
  } catch (error) {
    console.warn("Venus could not install the control indicator", error);
  }
}

async function removeControlIndicator(session, tabId) {
  const identifier = session.controlIndicators.get(tabId);
  session.controlIndicators.delete(tabId);
  if (identifier) {
    await sendCdp(tabId, "Page.removeScriptToEvaluateOnNewDocument", { identifier }).catch(() => {});
  }
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      await sendCdp(tabId, "Runtime.evaluate", {
        expression: `document.getElementById('__venus_control_indicator__')?.remove()`,
      });
      return;
    } catch {
      if (attempt < 2) await sleep(75);
    }
  }
}

async function renewControlIndicatorLeases(session) {
  const leaseUntil = Date.now() + 7_000;
  await Promise.allSettled([...session.attachedTabs].map((tabId) => sendCdp(tabId, "Runtime.evaluate", {
    expression: `(() => {
      const indicator = document.getElementById('__venus_control_indicator__');
      if (indicator) indicator.dataset.venusLeaseUntil = ${leaseUntil};
    })()`,
  })));
}

async function setPageAssistantEnabled(tabId, visible) {
  await chrome.tabs.sendMessage(tabId, {
    type: "venus_page_assistant_visible",
    visible,
  }).catch(() => {});
}

async function setControlIndicatorVisible(tabId, visible) {
  await chrome.tabs.sendMessage(tabId, {
    type: "venus_page_assistant_capture_visibility",
    visible,
  }).catch(() => {});
}

async function setControlIndicatorInteractive(tabId, interactive) {
  await chrome.tabs.sendMessage(tabId, {
    type: "venus_page_assistant_interactive",
    interactive,
  }).catch(() => {});
}

async function updateControlIndicator(session, payload = {}) {
  session.controlState = normalizeControlState(payload, session.controlState);
  const tabId = session.assistantTabId ?? session.activeTabId;
  if (Number.isInteger(tabId)) {
    await sendPageAssistantUpdate(tabId, session.controlState);
  }
  return { updated: Number.isInteger(tabId) };
}

async function refreshControlIndicatorScripts(session) {
  const source = controlIndicatorSource(
    session.controlState,
    session.controlBinding,
    session.controlToken,
  );
  await Promise.allSettled([...session.attachedTabs].map(async (tabId) => {
    const identifier = session.controlIndicators.get(tabId);
    if (identifier) {
      await sendCdp(tabId, "Page.removeScriptToEvaluateOnNewDocument", { identifier }).catch(() => {});
    }
    const installed = await sendCdp(tabId, "Page.addScriptToEvaluateOnNewDocument", { source });
    if (installed.identifier) session.controlIndicators.set(tabId, installed.identifier);
  }));
}

function normalizeControlState(payload, previous = {}) {
  const mode = ["running", "starting", "complete", "ready"].includes(payload?.mode)
    ? payload.mode
    : previous.mode || "running";
  return {
    mode,
    task: compactControlText(payload?.task ?? previous.task, 90),
    think: compactControlText(payload?.think ?? previous.think, 110),
    outcome: payload?.outcome === "call_user" ? "call_user" : "finished",
    content: String(payload?.content ?? "").trim(),
    notice: compactControlText(payload?.notice, 180),
    noticeTimeoutMs: Math.max(0, Math.min(10_000, Number(payload?.noticeTimeoutMs) || 0)),
  };
}

function compactControlText(value, maxLength) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function controlIndicatorSource(initialState, controlBinding, controlToken) {
  const state = normalizeControlState(initialState);
  return `(() => {
    if (globalThis.top !== globalThis) return;
    const hostId = '__venus_control_indicator__';
    const mount = () => {
      document.getElementById(hostId)?.remove();
      if (!document.documentElement) return;
      const host = document.createElement('div');
      host.id = hostId;
      host.dataset.venusLeaseUntil = String(Date.now() + 7000);
      host.dataset.interactive = 'true';
      host.dataset.mode = 'running';
      host.style.cssText = 'all:initial;position:fixed;inset:0;z-index:2147483647;pointer-events:none;display:block;';
      const shadow = host.attachShadow({ mode: 'closed' });
      shadow.innerHTML = \`
        <style>
          :host { all: initial; }
          .venus-layer { position: fixed; inset: 0; pointer-events: none; overflow: hidden; }
          .venus-grid {
            position: absolute; inset: 0;
            background-image: radial-gradient(circle, rgb(111 145 255 / 34%) 1px, transparent 1.25px);
            background-size: 11px 11px;
            opacity: .19;
            animation: venus-grid-breathe 2.8s ease-in-out infinite;
          }
          .venus-frame {
            position: absolute; inset: 0;
            border: 3px solid rgb(103 139 255 / 62%);
            border-radius: 8px;
            box-shadow: inset 0 0 24px rgb(88 119 255 / 34%), inset 0 0 70px rgb(112 76 220 / 16%);
            animation: venus-frame-pulse 1.8s ease-in-out infinite;
          }
          .venus-shell {
            position: absolute; left: 50%; bottom: 20px; transform: translateX(-50%);
            display: flex; align-items: flex-end; gap: 10px;
            width: max-content; max-width: calc(100vw - 32px);
          }
          .venus-badge {
            width: min(440px, calc(100vw - 32px)); box-sizing: border-box;
            padding: 9px 12px;
            border: 1px solid rgb(255 255 255 / 17%); border-radius: 18px;
            color: #fff; background: rgb(24 27 39 / 88%);
            box-shadow: 0 12px 38px rgb(20 24 48 / 36%), inset 0 1px rgb(255 255 255 / 10%);
            backdrop-filter: blur(13px); -webkit-backdrop-filter: blur(13px);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          }
          .venus-status-row { display: flex; align-items: center; gap: 12px; min-height: 36px; }
          .venus-mark {
            all: initial; position: relative; display: block; width: 30px; height: 30px; flex: 0 0 30px;
            border-radius: 50%; cursor: pointer; pointer-events: auto;
            background: rgb(145 169 255 / 10%);
            transition: width .32s cubic-bezier(.2,.8,.2,1), height .32s cubic-bezier(.2,.8,.2,1),
              border-radius .32s cubic-bezier(.2,.8,.2,1), background .25s ease, transform .25s ease;
          }
          .venus-mark::before, .venus-mark::after {
            content: ''; position: absolute; inset: 6px; border: 2px solid #bca8ff; border-radius: 50%;
            transition: inset .32s cubic-bezier(.2,.8,.2,1), border-radius .32s cubic-bezier(.2,.8,.2,1);
          }
          .venus-mark::after { inset: 11px; border-color: #fff; }
          .venus-mark:hover, .venus-mark:focus-visible {
            border-radius: 8px; background: rgb(145 169 255 / 24%); transform: scale(1.08);
          }
          .venus-mark:hover::before, .venus-mark:focus-visible::before { inset: 5px; border-radius: 4px; }
          .venus-mark:hover::after, .venus-mark:focus-visible::after { inset: 10px; border-radius: 2px; }
          .venus-mark:active { transform: scale(.94); }
          :host([data-interactive='false']) .venus-mark { pointer-events: none; }
          :host(:not([data-mode='running'])) .venus-mark { display: none; }
          .venus-copy { min-width: 0; }
          .venus-task { overflow: hidden; color: #fff; font-size: 13px; font-weight: 700; line-height: 1.35; text-overflow: ellipsis; white-space: nowrap; }
          .venus-state { display: flex; align-items: center; gap: 6px; margin-top: 3px; color: #c9d3ff; font-size: 11px; font-weight: 600; }
          .venus-dot { width: 6px; height: 6px; border-radius: 50%; background: #91a9ff; box-shadow: 0 0 9px #91a9ff; animation: venus-dot 1.2s ease-in-out infinite; }
          .venus-composer { display: flex; align-items: flex-end; gap: 8px; margin-top: 8px; }
          .venus-input {
            all: initial; display: block; width: 100%; min-width: 0; height: 36px; max-height: 92px;
            box-sizing: border-box; resize: none; overflow-y: auto; padding: 8px 11px;
            border: 1px solid rgb(173 188 255 / 28%); border-radius: 12px;
            color: #fff; background: rgb(255 255 255 / 8%); caret-color: #fff;
            font: 13px/18px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            pointer-events: auto;
          }
          .venus-input::placeholder { color: rgb(215 221 255 / 58%); }
          .venus-input:focus { border-color: rgb(162 179 255 / 68%); box-shadow: 0 0 0 2px rgb(116 139 255 / 18%); }
          .venus-submit {
            all: initial; display: grid; place-items: center; width: 36px; height: 36px; flex: 0 0 36px;
            border-radius: 50%; color: #fff; background: linear-gradient(145deg, #8e7cff, #668cff);
            box-shadow: 0 5px 16px rgb(92 112 255 / 38%); cursor: pointer; pointer-events: auto;
            font: 700 21px/1 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            transition: transform .18s ease, filter .18s ease;
          }
          .venus-submit:hover, .venus-submit:focus-visible { transform: scale(1.06); filter: brightness(1.08); }
          .venus-submit:active { transform: scale(.94); }
          .venus-submit:disabled, .venus-input:disabled { opacity: .55; cursor: default; }
          .venus-result {
            position: relative; width: min(280px, calc(100vw - 32px)); max-height: 180px; box-sizing: border-box;
            overflow: auto; padding: 12px 34px 12px 14px; border: 1px solid rgb(180 192 255 / 28%);
            border-radius: 15px; color: #eef1ff; background: rgb(31 34 48 / 94%);
            box-shadow: 0 12px 36px rgb(20 24 48 / 34%); backdrop-filter: blur(13px);
            -webkit-backdrop-filter: blur(13px); pointer-events: auto;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          }
          .venus-result-title { color: #b9c6ff; font-size: 11px; font-weight: 750; }
          .venus-result-content { margin: 5px 0 0; color: #fff; font-size: 13px; line-height: 1.45; white-space: pre-wrap; }
          .venus-result-close {
            all: initial; position: absolute; top: 7px; right: 9px; display: grid; place-items: center;
            width: 22px; height: 22px; border-radius: 50%; color: #cbd3f5; cursor: pointer;
            pointer-events: auto; font: 18px/1 sans-serif;
          }
          .venus-result-close:hover, .venus-result-close:focus-visible { color: #fff; background: rgb(255 255 255 / 10%); }
          [hidden] { display: none !important; }
          :host(:not([data-mode='running'])) .venus-grid,
          :host(:not([data-mode='running'])) .venus-frame { display: none; }
          @keyframes venus-grid-breathe { 50% { opacity: .1; } }
          @keyframes venus-frame-pulse { 50% { border-color: rgb(150 119 255 / 78%); box-shadow: inset 0 0 34px rgb(100 135 255 / 42%), inset 0 0 85px rgb(112 76 220 / 20%); } }
          @keyframes venus-dot { 50% { opacity: .35; transform: scale(.72); } }
          @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
        </style>
        <div class="venus-layer">
          <div class="venus-grid"></div>
          <div class="venus-frame"></div>
          <div class="venus-shell">
            <div class="venus-badge">
              <div class="venus-status-row">
                <button class="venus-mark" type="button" aria-label="停止 Venus Agent" title="停止任务"></button>
                <span class="venus-copy">
                  <span class="venus-task"></span>
                  <span class="venus-state"><span class="venus-dot"></span><span class="venus-state-label"></span></span>
                </span>
              </div>
              <form class="venus-composer" hidden>
                <textarea class="venus-input" rows="1" placeholder="输入新任务…" aria-label="任务"></textarea>
                <button class="venus-submit" type="submit" aria-label="运行任务" title="运行任务">→</button>
              </form>
            </div>
            <aside class="venus-result" aria-live="polite" hidden>
              <button class="venus-result-close" type="button" aria-label="关闭结果">×</button>
              <div class="venus-result-title"></div>
              <p class="venus-result-content"></p>
            </aside>
          </div>
        </div>\`;
      const taskLabel = shadow.querySelector('.venus-task');
      const stateLabel = shadow.querySelector('.venus-state-label');
      const composer = shadow.querySelector('.venus-composer');
      const input = shadow.querySelector('.venus-input');
      const submitButton = shadow.querySelector('.venus-submit');
      const result = shadow.querySelector('.venus-result');
      const resultTitle = shadow.querySelector('.venus-result-title');
      const resultContent = shadow.querySelector('.venus-result-content');
      const stopButton = shadow.querySelector('.venus-mark');
      const isolatedEvents = [
        'keydown', 'keyup', 'keypress',
        'beforeinput', 'input', 'change',
        'compositionstart', 'compositionupdate', 'compositionend',
        'paste', 'copy', 'cut',
        'click', 'dblclick', 'mousedown', 'mouseup',
        'pointerdown', 'pointerup', 'touchstart', 'touchend',
      ];
      for (const eventName of isolatedEvents) {
        shadow.addEventListener(eventName, (event) => event.stopPropagation());
      }
      const sendControl = (payload) => {
        const binding = globalThis[${JSON.stringify(controlBinding)}];
        if (typeof binding === 'function') binding(JSON.stringify({
          ...payload,
          token: ${JSON.stringify(controlToken)},
        }));
      };
      const applyState = (nextState = {}) => {
        const mode = ['running', 'starting', 'complete', 'ready'].includes(nextState.mode)
          ? nextState.mode
          : 'running';
        host.dataset.mode = mode;
        host.dataset.interactive = mode === 'running' ? 'true' : 'false';
        taskLabel.textContent = nextState.task || (mode === 'ready' ? 'Venus Browser Agent' : '正在执行浏览器任务');
        stateLabel.textContent = nextState.think || (mode === 'starting' ? '正在启动…' : '就绪');
        composer.hidden = mode === 'running' || mode === 'starting';
        input.disabled = mode === 'starting';
        submitButton.disabled = mode === 'starting';
        stopButton.disabled = mode !== 'running';
        if (mode === 'complete' && nextState.content) {
          result.hidden = false;
          resultTitle.textContent = nextState.outcome === 'call_user' ? '需要用户处理' : '任务完成';
          resultContent.textContent = nextState.content;
        } else if (mode === 'ready' && nextState.notice) {
          result.hidden = false;
          resultTitle.textContent = '无法运行任务';
          resultContent.textContent = nextState.notice;
        } else if (mode === 'running' || mode === 'starting') {
          result.hidden = true;
        }
      };
      host.addEventListener('__venus_control_update__', (event) => applyState(event.detail));
      stopButton.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (host.dataset.interactive !== 'true' || stopButton.disabled) return;
        stopButton.disabled = true;
        stateLabel.textContent = '正在停止…';
        sendControl({ type: 'stop' });
      });
      composer.addEventListener('submit', (event) => {
        event.preventDefault();
        const task = input.value.trim();
        if (!task || submitButton.disabled) {
          input.focus();
          return;
        }
        applyState({ mode: 'starting', task, think: '正在启动…' });
        sendControl({ type: 'run_task', task });
      });
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
          event.preventDefault();
          composer.requestSubmit();
        }
      });
      shadow.querySelector('.venus-result-close').addEventListener('click', () => {
        result.hidden = true;
      });
      applyState(${JSON.stringify(state)});
      document.documentElement.append(host);
      const watchdog = setInterval(() => {
        if (!host.isConnected || Date.now() > Number(host.dataset.venusLeaseUntil || 0)) {
          host.remove();
          clearInterval(watchdog);
        }
      }, 1000);
    };
    if (document.documentElement) mount();
    else addEventListener('DOMContentLoaded', mount, { once: true });
  })()`;
}

async function captureObservation(session) {
  if (session.bootstrapTabId && session.attachedTabs.size === 0) {
    return captureBootstrapObservation(session);
  }
  await adoptAgentOpenedTab(session);
  const tabId = await requireActiveTab(session);
  const tab = await waitForControllableTab(tabId);

  await setControlIndicatorVisible(tabId, false);
  let metrics;
  let screenshot;
  try {
    [metrics, screenshot] = await Promise.all([
      sendCdp(tabId, "Page.getLayoutMetrics"),
      sendCdp(tabId, "Page.captureScreenshot", {
        format: "jpeg",
        quality: 82,
        fromSurface: true,
        captureBeyondViewport: false,
      }),
    ]);
  } finally {
    await setControlIndicatorVisible(tabId, true);
  }

  const viewport = metrics.cssVisualViewport ?? metrics.cssLayoutViewport ?? metrics.layoutViewport;
  const width = Number(viewport?.clientWidth);
  const height = Number(viewport?.clientHeight);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error("无法读取页面 viewport 尺寸");
  }
  session.viewport = { width, height };

  const fileTransfers = session.pendingTransferNotices.splice(0);
  return {
    tab: tabSummary(tab),
    viewport: session.viewport,
    screenshot: screenshot.data,
    fileTransfers,
    capturedAt: new Date().toISOString(),
  };
}

async function captureBootstrapObservation(session) {
  const tab = await chrome.tabs.get(session.bootstrapTabId);
  if (isControllableUrl(tab.url)) {
    const tabId = session.bootstrapTabId;
    session.bootstrapTabId = null;
    await attachTab(session, tabId);
    session.activeTabId = tabId;
    await bindPageAssistantToTab(session, tab);
    return captureObservation(session);
  }
  if (!isBootstrapUrl(tab.url || tab.pendingUrl)) {
    assertControllableUrl(tab.url);
  }

  const windowInfo = await chrome.windows.get(tab.windowId).catch(() => null);
  const width = Math.max(1, Number(windowInfo?.width) || 1200);
  const height = Math.max(1, Number(windowInfo?.height) || 800);
  let screenshot;
  try {
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
      format: "jpeg",
      quality: 82,
    });
    screenshot = String(dataUrl).replace(/^data:image\/jpeg;base64,/, "");
  } catch (error) {
    console.warn("Venus could not capture the protected new-tab page; using a placeholder", error);
    screenshot = await createBootstrapScreenshot(width, height);
  }
  return {
    tab: tabSummary(tab),
    viewport: { width, height },
    screenshot,
    fileTransfers: [],
    capturedAt: new Date().toISOString(),
    bootstrap: true,
  };
}

async function createBootstrapScreenshot(width, height) {
  if (typeof OffscreenCanvas !== "function") {
    throw new Error("当前 Chrome 无法截图新标签页；请先打开任意网页再运行任务");
  }
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d");
  context.fillStyle = "#f5f3ee";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#6f45d2";
  context.font = "700 28px sans-serif";
  context.textAlign = "center";
  context.fillText("Venus Browser Agent", width / 2, height / 2 - 12);
  context.fillStyle = "#726f66";
  context.font = "16px sans-serif";
  context.fillText("Blank new tab — use Launch(url='https://...') to begin", width / 2, height / 2 + 24);
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.82 });
  return bytesToBase64(new Uint8Array(await blob.arrayBuffer()));
}

async function executeAction(session, action) {
  if (!action?.name) {
    throw new Error("Action 缺少 name");
  }
  if (action.name === "launch") {
    const result = await launch(session, action.url);
    await sleep(900);
    return result;
  }
  const tabId = await requireActiveTab(session);
  if (!session.viewport) {
    throw new Error("执行 action 前必须先截图");
  }

  await setControlIndicatorInteractive(tabId, false);
  try {
    let result = { action: action.name };
    switch (action.name) {
    case "click": {
      const interaction = beginClickInteraction(session, tabId);
      try {
        const beforeClick = await chrome.tabs.get(tabId).catch(() => null);
        result = await clickAt(session, tabId, action.point, 1);
        if (result.downloadTarget?.intercepted) {
          await sleep(250);
        } else {
          result.page = await settlePageAfterClick(tabId, beforeClick?.url || "");
          if (shouldFallbackToLinkNavigation(result.clickTarget, result.page.url)) {
            await sendCdp(tabId, "Page.navigate", { url: result.clickTarget.url });
            result.page = await settlePageAfterClick(tabId, beforeClick?.url || "");
            result.page.fallback = "direct_link_navigation";
          }
        }
        await settleInteraction(interaction);
        if (interaction.events.length) {
          result.fileTransfers = interaction.events;
        }
        await adoptAgentOpenedTab(session);
      } finally {
        if (session.clickInteraction === interaction) {
          session.clickInteraction = null;
        }
      }
      break;
    }
    case "double_click":
      result = await clickAt(session, tabId, action.point, 2);
      await sleep(500);
      break;
    case "hover":
      await moveMouse(tabId, toCssPoint(session, action.point));
      await sleep(350);
      break;
    case "long_press":
      await longPress(session, tabId, action);
      break;
    case "drag":
      await drag(session, tabId, action.start, action.end);
      break;
    case "scroll":
      await scroll(session, tabId, action);
      break;
    case "type":
      await sendCdp(tabId, "Input.insertText", { text: String(action.content ?? "") });
      await sleep(300);
      break;
    case "upload":
      result = await uploadWorkspaceFile(session, action.file);
      break;
    case "download":
      result = await finalizeDownload(session, action.filename);
      break;
    case "hotkey":
      result = await hotkey(session, tabId, action);
      break;
    case "press_enter":
      await dispatchKey(tabId, "Enter", 0);
      await sleep(500);
      break;
    case "press_home":
      await dispatchKey(tabId, "Home", 0);
      await sleep(300);
      break;
    case "press_back":
      await navigateBack(tabId);
      await sleep(900);
      break;
    case "wait":
      await sleep(2500);
      break;
    case "get_url": {
      const tab = await chrome.tabs.get(tabId);
      result = { action: action.name, url: tab.url ?? "" };
      break;
    }
    case "select_option":
      result = await selectOption(session, tabId, action);
      break;
    case "take_note":
      result = { action: action.name, note: String(action.content ?? "") };
      break;
      default:
        throw new Error(`浏览器执行器不支持 action：${action.name}`);
    }
    return result;
  } finally {
    if (!session.stopRequested && session.attachedTabs.has(tabId)) {
      await setControlIndicatorInteractive(tabId, true);
    }
  }
}

async function clickAt(session, tabId, normalizedPoint, clickCount) {
  const point = toCssPoint(session, normalizedPoint);
  const selectContext = clickCount === 1 ? await inspectSelect(tabId, point) : null;
  const inspectedTarget = clickCount === 1
    ? await inspectDownloadTarget(tabId, point)
    : null;
  let downloadTarget = null;
  if (isLikelyDownloadLink(inspectedTarget)) {
    const protocol = safeUrlProtocol(inspectedTarget.url);
    if (["http:", "https:", "blob:", "data:"].includes(protocol)) {
      downloadTarget = queueRemoteDownload(session, tabId, inspectedTarget);
      return {
        action: "click",
        selectContext,
        clickTarget: publicClickTarget(inspectedTarget),
        downloadTarget,
      };
    }
    downloadTarget = {
      url: inspectedTarget.url,
      text: inspectedTarget.text,
      mimeType: inspectedTarget.mimeType,
      forced: await forceDownloadTarget(tabId, inspectedTarget.marker),
    };
  }
  await moveMouse(tabId, point);
  await sendCdp(tabId, "Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: point.x,
    y: point.y,
    button: "left",
    buttons: 1,
    clickCount,
  });
  await sendCdp(tabId, "Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x: point.x,
    y: point.y,
    button: "left",
    buttons: 0,
    clickCount,
  });
  if (selectContext) {
    session.selectContexts.set(tabId, selectContext);
  } else {
    session.selectContexts.delete(tabId);
  }
  return {
    action: clickCount === 2 ? "double_click" : "click",
    selectContext,
    ...(inspectedTarget ? { clickTarget: publicClickTarget(inspectedTarget) } : {}),
    ...(downloadTarget ? { downloadTarget } : {}),
  };
}

function publicClickTarget(target) {
  return {
    url: target?.url || "",
    text: target?.text || "",
    mimeType: target?.mimeType || "",
    hasDownloadAttribute: Boolean(target?.hasDownloadAttribute),
  };
}

function shouldFallbackToLinkNavigation(target, currentUrl) {
  if (!target?.url || isLikelyDownloadLink(target)) return false;
  if (!['http:', 'https:'].includes(safeUrlProtocol(target.url))) return false;
  try {
    const targetUrl = new URL(target.url);
    const current = new URL(currentUrl);
    return targetUrl.href !== current.href;
  } catch {
    return false;
  }
}

async function settlePageAfterClick(tabId, initialUrl, timeoutMs = 3500) {
  const startedAt = Date.now();
  const deadline = startedAt + timeoutMs;
  let changed = false;
  let stableSince = 0;
  let latest = null;

  await sleep(250);
  while (Date.now() < deadline) {
    try {
      latest = await chrome.tabs.get(tabId);
    } catch {
      break;
    }
    changed ||= Boolean(latest.url && latest.url !== initialUrl);
    if (changed && latest.status === "complete") {
      stableSince ||= Date.now();
      if (Date.now() - stableSince >= 300) break;
    } else {
      stableSince = 0;
    }
    // Preserve the old latency for clicks that do not navigate.
    if (!changed && latest.status !== "loading" && Date.now() - startedAt >= 700) break;
    await sleep(100);
  }
  return {
    urlChanged: changed,
    url: latest?.url || initialUrl,
    status: latest?.status || "",
  };
}

async function moveMouse(tabId, point) {
  await sendCdp(tabId, "Input.dispatchMouseEvent", {
    type: "mouseMoved",
    x: point.x,
    y: point.y,
    button: "none",
  });
}

async function longPress(session, tabId, action) {
  const point = toCssPoint(session, action.point);
  const duration = Math.min(Math.max(Number(action.duration) || 20, 0.05), 30);
  await moveMouse(tabId, point);
  await sendCdp(tabId, "Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: point.x,
    y: point.y,
    button: "left",
    buttons: 1,
    clickCount: 1,
  });
  try {
    await sleep(duration * 1000);
  } finally {
    await sendCdp(tabId, "Input.dispatchMouseEvent", {
      type: "mouseReleased",
      x: point.x,
      y: point.y,
      button: "left",
      buttons: 0,
      clickCount: 1,
    });
  }
}

async function drag(session, tabId, startPoint, endPoint) {
  const start = toCssPoint(session, startPoint);
  const end = toCssPoint(session, endPoint);
  await moveMouse(tabId, start);
  await sendCdp(tabId, "Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: start.x,
    y: start.y,
    button: "left",
    buttons: 1,
    clickCount: 1,
  });
  try {
    for (let step = 1; step <= 12; step += 1) {
      const ratio = step / 12;
      await sendCdp(tabId, "Input.dispatchMouseEvent", {
        type: "mouseMoved",
        x: start.x + (end.x - start.x) * ratio,
        y: start.y + (end.y - start.y) * ratio,
        button: "left",
        buttons: 1,
      });
      await sleep(20);
    }
  } finally {
    await sendCdp(tabId, "Input.dispatchMouseEvent", {
      type: "mouseReleased",
      x: end.x,
      y: end.y,
      button: "left",
      buttons: 0,
      clickCount: 1,
    });
  }
  await sleep(300);
}

async function scroll(session, tabId, action) {
  const point = toCssPoint(session, action.point);
  const amount = Math.round(session.viewport.height * 0.66);
  const deltas = {
    up: [0, -amount],
    down: [0, amount],
    left: [-amount, 0],
    right: [amount, 0],
  };
  const [deltaX, deltaY] = deltas[action.direction] ?? deltas.down;
  await sendCdp(tabId, "Input.dispatchMouseEvent", {
    type: "mouseWheel",
    x: point.x,
    y: point.y,
    deltaX,
    deltaY,
  });
  await sleep(450);
}

async function hotkey(session, tabId, action) {
  const keys = action.keys.map((key) => String(key).toLowerCase());
  const hasTab = keys.includes("tab");
  const hasSwitchModifier = keys.some((key) => ["ctrl", "control", "meta", "cmd", "controlormeta"].includes(key));
  if (hasTab && hasSwitchModifier) {
    const direction = keys.includes("shift") ? -1 : 1;
    for (let repeat = 0; repeat < action.repeat; repeat += 1) {
      await switchAgentTab(session, direction);
    }
    return { action: "hotkey", switchedTab: session.activeTabId };
  }

  const platform = await chrome.runtime.getPlatformInfo();
  const normalized = normalizeHotkeyKeys(keys, platform.os);
  let modifiers = 0;
  if (normalized.includes("alt")) modifiers |= 1;
  if (normalized.includes("ctrl")) modifiers |= 2;
  if (normalized.includes("meta")) modifiers |= 4;
  if (normalized.includes("shift")) modifiers |= 8;
  const primary = normalized.find((key) => !["alt", "ctrl", "meta", "shift"].includes(key));
  if (!primary) {
    throw new Error("Hotkey 缺少非修饰键");
  }
  const modifierKeys = normalized.filter((key, index) =>
    ["alt", "ctrl", "meta", "shift"].includes(key) && normalized.indexOf(key) === index,
  );
  const commands = editingCommandsForHotkey(normalized, platform.os);
  for (let repeat = 0; repeat < action.repeat; repeat += 1) {
    await dispatchKey(tabId, primary, modifiers, modifierKeys, commands);
  }
  await sleep(250);
  return { action: "hotkey" };
}

async function dispatchKey(tabId, keyName, modifiers, modifierKeys = [], commands = []) {
  const descriptor = keyDescriptor(keyName);
  let heldModifiers = 0;
  for (const modifierKey of modifierKeys) {
    heldModifiers |= modifierBit(modifierKey);
    await sendCdp(tabId, "Input.dispatchKeyEvent", {
      type: "rawKeyDown",
      modifiers: heldModifiers,
      ...keyDescriptor(modifierKey),
    });
  }
  await sendCdp(tabId, "Input.dispatchKeyEvent", {
    type: "rawKeyDown",
    modifiers,
    ...descriptor,
    ...(commands.length > 0 ? { commands } : {}),
  });
  await sendCdp(tabId, "Input.dispatchKeyEvent", {
    type: "keyUp",
    modifiers,
    ...descriptor,
  });
  for (const modifierKey of [...modifierKeys].reverse()) {
    heldModifiers &= ~modifierBit(modifierKey);
    await sendCdp(tabId, "Input.dispatchKeyEvent", {
      type: "keyUp",
      modifiers: heldModifiers,
      ...keyDescriptor(modifierKey),
    });
  }
}

function modifierBit(keyName) {
  return { alt: 1, ctrl: 2, meta: 4, shift: 8 }[keyName] ?? 0;
}

function keyDescriptor(keyName) {
  const normalized = String(keyName).toLowerCase();
  const special = {
    alt: { key: "Alt", code: "AltLeft", windowsVirtualKeyCode: 18 },
    ctrl: { key: "Control", code: "ControlLeft", windowsVirtualKeyCode: 17 },
    meta: { key: "Meta", code: "MetaLeft", windowsVirtualKeyCode: 91 },
    shift: { key: "Shift", code: "ShiftLeft", windowsVirtualKeyCode: 16 },
    enter: { key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 },
    tab: { key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 },
    escape: { key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 },
    esc: { key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 },
    backspace: { key: "Backspace", code: "Backspace", windowsVirtualKeyCode: 8 },
    delete: { key: "Delete", code: "Delete", windowsVirtualKeyCode: 46 },
    home: { key: "Home", code: "Home", windowsVirtualKeyCode: 36 },
    arrowup: { key: "ArrowUp", code: "ArrowUp", windowsVirtualKeyCode: 38 },
    arrowdown: { key: "ArrowDown", code: "ArrowDown", windowsVirtualKeyCode: 40 },
    arrowleft: { key: "ArrowLeft", code: "ArrowLeft", windowsVirtualKeyCode: 37 },
    arrowright: { key: "ArrowRight", code: "ArrowRight", windowsVirtualKeyCode: 39 },
    space: { key: " ", code: "Space", windowsVirtualKeyCode: 32 },
  };
  if (special[normalized]) {
    return special[normalized];
  }
  if (normalized.length === 1) {
    const upper = normalized.toUpperCase();
    return {
      key: normalized,
      code: /[a-z]/.test(normalized) ? `Key${upper}` : normalized,
      windowsVirtualKeyCode: upper.charCodeAt(0),
    };
  }
  return { key: keyName, code: keyName, windowsVirtualKeyCode: 0 };
}

async function navigateBack(tabId) {
  const history = await sendCdp(tabId, "Page.getNavigationHistory");
  const targetIndex = history.currentIndex - 1;
  if (targetIndex < 0 || !history.entries?.[targetIndex]) {
    return;
  }
  await sendCdp(tabId, "Page.navigateToHistoryEntry", {
    entryId: history.entries[targetIndex].id,
  });
}

async function launch(session, rawUrl) {
  const url = new URL(rawUrl);
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error("Launch 仅允许 http/https URL");
  }
  let createdTab;
  if (session.bootstrapTabId) {
    const bootstrapTabId = session.bootstrapTabId;
    session.bootstrapTabId = null;
    createdTab = await chrome.tabs.update(bootstrapTabId, {
      url: url.href,
      active: true,
    });
  } else {
    createdTab = await chrome.tabs.create({
      windowId: session.windowId ?? undefined,
      url: url.href,
      active: true,
    });
  }
  const tab = await waitForControllableTab(createdTab.id, 20_000);
  await attachTab(session, tab.id);
  session.activeTabId = tab.id;
  session.windowId = tab.windowId;
  await bindPageAssistantToTab(session, tab);
  return { action: "launch", tab: tabSummary(tab) };
}

async function switchAgentTab(session, direction) {
  const tabs = await chrome.tabs.query({ windowId: session.windowId });
  const ownedTabs = tabs.filter((tab) => session.attachedTabs.has(tab.id));
  if (ownedTabs.length < 2) {
    return;
  }
  let index = ownedTabs.findIndex((tab) => tab.id === session.activeTabId);
  if (index < 0) index = 0;
  index = (index + direction + ownedTabs.length) % ownedTabs.length;
  const target = ownedTabs[index];
  await chrome.tabs.update(target.id, { active: true });
  session.activeTabId = target.id;
  await bindPageAssistantToTab(session, target);
  session.viewport = null;
}

async function inspectSelect(tabId, point) {
  const marker = `venus-select-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const expression = `(() => {
    const element = document.elementFromPoint(${JSON.stringify(point.x)}, ${JSON.stringify(point.y)});
    const select = element && (element.matches?.('select') ? element : element.closest?.('select'));
    if (!select) return null;
    select.dataset.venusSelectId = ${JSON.stringify(marker)};
    return {
      marker: ${JSON.stringify(marker)},
      name: select.getAttribute('aria-label') || select.name || select.id || '',
      multiple: Boolean(select.multiple),
      options: Array.from(select.options).map((option, index) => ({
        index,
        text: option.text,
        value: option.value,
        selected: option.selected,
        disabled: option.disabled
      }))
    };
  })()`;
  try {
    const result = await sendCdp(tabId, "Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: false,
    });
    return result.result?.value ?? null;
  } catch {
    return null;
  }
}

async function inspectDownloadTarget(tabId, point) {
  const marker = `venus-download-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  let visibleTabUrl = "";
  try {
    visibleTabUrl = (await chrome.tabs.get(tabId)).url || "";
  } catch {
    // The tab can be replaced while a PDF document is loading.
  }
  const expression = `(() => {
    const x = ${JSON.stringify(point.x)};
    const y = ${JSON.stringify(point.y)};
    let element = document.elementFromPoint(x, y);
    const visited = new Set();
    while (element?.shadowRoot && !visited.has(element)) {
      visited.add(element);
      const nested = element.shadowRoot.elementFromPoint(x, y);
      if (!nested || nested === element) break;
      element = nested;
    }
    const distanceToRect = (candidate) => {
      const rect = candidate.getBoundingClientRect?.();
      if (!rect || rect.width <= 0 || rect.height <= 0) return Infinity;
      const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
      const dy = y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
      return Math.hypot(dx, dy);
    };
    const nearbyClickable = Array.from(document.querySelectorAll(
      'a[href], area[href], button, [role="button"], cr-icon-button, viewer-download-controls'
    ))
      .map((candidate) => ({ candidate, distance: distanceToRect(candidate) }))
      .filter((item) => item.distance <= 32)
      .sort((left, right) => left.distance - right.distance)[0]?.candidate || null;
    if (!element && !nearbyClickable) return null;

    const path = [];
    let current = element;
    while (current) {
      path.push(current);
      current = current.parentElement || current.getRootNode?.().host || null;
    }
    const anchor = path.find((item) => item.matches?.('a[href], area[href]'))
      || (nearbyClickable?.matches?.('a[href], area[href]') ? nearbyClickable : null);
    if (anchor) {
      anchor.dataset.venusDownloadId = ${JSON.stringify(marker)};
      return {
        marker: ${JSON.stringify(marker)},
        url: anchor.href || '',
        text: (anchor.innerText || anchor.textContent || anchor.getAttribute('aria-label') || '').trim(),
        mimeType: anchor.getAttribute('type') || '',
        hasDownloadAttribute: anchor.hasAttribute('download')
      };
    }

    const control = path.find((item) => item.matches?.(
      'button, [role="button"], cr-icon-button, viewer-download-controls'
    )) || (nearbyClickable?.matches?.(
      'button, [role="button"], cr-icon-button, viewer-download-controls'
    ) ? nearbyClickable : null) || element;
    const description = [
      control.innerText,
      control.textContent,
      control.getAttribute?.('aria-label'),
      control.getAttribute?.('title'),
      control.getAttribute?.('data-tooltip-text'),
      control.id,
      control.getAttribute?.('icon'),
    ].filter(Boolean).join(' ').trim();
    const looksLikeDownload = /(?:download|save|下载|保存)/i.test(description);
    if (looksLikeDownload && location.hostname === 'github.com') {
      const rawPath = location.pathname.replace('/blob/', '/raw/');
      if (rawPath !== location.pathname) {
        return {
          marker: '',
          url: new URL(rawPath + location.search, location.origin).href,
          text: description || 'Download raw file',
          mimeType: 'application/octet-stream',
          hasDownloadAttribute: true
        };
      }
    }
    const embeddedPdf = document.querySelector('embed[type="application/pdf"]');
    const currentUrl = String(
      embeddedPdf?.getAttribute('original-url')
      || ${JSON.stringify(visibleTabUrl)}
      || location.href
      || ''
    );
    let isPdfDocument = document.contentType === 'application/pdf';
    try {
      const parsed = new URL(currentUrl);
      isPdfDocument ||= /\\.pdf$/i.test(parsed.pathname) || /\\/(?:pdf|download)\\//i.test(parsed.pathname);
    } catch {}
    if (!looksLikeDownload || !isPdfDocument || !/^https?:/i.test(currentUrl)) return null;

    return {
      marker: '',
      url: currentUrl,
      text: description || 'Download PDF',
      mimeType: 'application/pdf',
      hasDownloadAttribute: true
    };
  })()`;
  try {
    const response = await sendCdp(tabId, "Runtime.evaluate", {
      expression,
      returnByValue: true,
    });
    const target = response.result?.value ?? null;
    if (target) return target;
  } catch {
    // Chrome's built-in PDF viewer UI is not always exposed to the page's
    // JavaScript world. Fall through to its accessibility tree.
  }
  return inspectPdfViewerDownloadButton(tabId, point, visibleTabUrl);
}

async function inspectPdfViewerDownloadButton(tabId, point, visibleTabUrl) {
  if (!isLikelyPdfDocumentUrl(visibleTabUrl)) return null;
  try {
    await sendCdp(tabId, "Accessibility.enable");
    const tree = await sendCdp(tabId, "Accessibility.getFullAXTree");
    const candidates = (tree.nodes ?? []).filter((node) => {
      const role = String(node.role?.value ?? "").toLowerCase();
      const description = [node.name?.value, node.description?.value]
        .filter(Boolean)
        .join(" ");
      return role === "button" && /(?:download|save|下载|保存)/i.test(description);
    });
    for (const node of candidates) {
      if (!Number.isInteger(node.backendDOMNodeId)) continue;
      let model;
      try {
        model = await sendCdp(tabId, "DOM.getBoxModel", {
          backendNodeId: node.backendDOMNodeId,
        });
      } catch {
        continue;
      }
      const quad = model.model?.border ?? model.model?.content;
      if (!pointInsideQuad(point, quad)) continue;
      return {
        marker: "",
        url: visibleTabUrl,
        text: String(node.name?.value || "Download PDF"),
        mimeType: "application/pdf",
        hasDownloadAttribute: true,
      };
    }
  } catch (error) {
    console.warn("Venus could not inspect the PDF viewer toolbar", error);
  }

  // Some Chrome builds expose the PDF toolbar visually and to mouse input,
  // but omit its internal extension nodes from the inspected target's AX
  // tree. The download control has a stable top-right toolbar position. Keep
  // this fallback deliberately narrow and PDF-only.
  try {
    const metrics = await sendCdp(tabId, "Page.getLayoutMetrics");
    const viewport = metrics.cssVisualViewport ?? metrics.cssLayoutViewport ?? metrics.layoutViewport;
    const width = Number(viewport?.clientWidth);
    const distanceFromRight = width - point.x;
    if (Number.isFinite(width)
      && point.y >= 0
      && point.y <= 72
      && distanceFromRight >= 95
      && distanceFromRight <= 155) {
      return {
        marker: "",
        url: visibleTabUrl,
        text: "Download PDF",
        mimeType: "application/pdf",
        hasDownloadAttribute: true,
      };
    }
  } catch {
    // If layout metrics are unavailable, preserve normal click behavior.
  }
  return null;
}

function pointInsideQuad(point, quad) {
  if (!Array.isArray(quad) || quad.length < 8) return false;
  const xs = [quad[0], quad[2], quad[4], quad[6]];
  const ys = [quad[1], quad[3], quad[5], quad[7]];
  return point.x >= Math.min(...xs)
    && point.x <= Math.max(...xs)
    && point.y >= Math.min(...ys)
    && point.y <= Math.max(...ys);
}

function isLikelyPdfDocumentUrl(rawUrl) {
  try {
    const pathname = new URL(rawUrl).pathname;
    return /\.pdf$/i.test(pathname) || /\/(?:pdf|download)\//i.test(pathname);
  } catch {
    return false;
  }
}

async function forceDownloadTarget(tabId, marker) {
  if (!marker) return false;
  const expression = `(() => {
    const marker = ${JSON.stringify(marker)};
    const anchor = Array.from(document.querySelectorAll('[data-venus-download-id]'))
      .find((item) => item.dataset.venusDownloadId === marker);
    if (!anchor) return false;
    const targetUrl = new URL(anchor.href, document.baseURI);
    if (!['http:', 'https:', 'blob:'].includes(targetUrl.protocol)) return false;
    if (targetUrl.protocol !== 'blob:' && targetUrl.origin !== location.origin) return false;
    anchor.setAttribute('download', '');
    anchor.removeAttribute('target');
    return true;
  })()`;
  try {
    const response = await sendCdp(tabId, "Runtime.evaluate", {
      expression,
      returnByValue: true,
    });
    return response.result?.value === true;
  } catch {
    return false;
  }
}

async function selectOption(session, tabId, action) {
  const context = session.selectContexts.get(tabId);
  if (!context?.marker) {
    throw new Error("没有可用的原生 select；请先点击 select 控件");
  }
  const args = JSON.stringify({
    marker: context.marker,
    index: action.index,
    value: action.value,
    text: action.text,
  });
  const expression = `((args) => {
    const select = document.querySelector('[data-venus-select-id="' + CSS.escape(args.marker) + '"]');
    if (!select) return { ok: false, error: 'select_not_found' };
    const options = Array.from(select.options);
    let option = null;
    if (Number.isInteger(args.index)) option = options[args.index] || null;
    if (!option && args.value !== undefined) option = options.find(item => item.value === args.value) || null;
    if (!option && args.text !== undefined) option = options.find(item => item.text === args.text) || null;
    if (!option || option.disabled) return { ok: false, error: 'option_not_found' };
    select.value = option.value;
    option.selected = true;
    select.dispatchEvent(new Event('input', { bubbles: true }));
    select.dispatchEvent(new Event('change', { bubbles: true }));
    return { ok: true, index: option.index, text: option.text, value: option.value };
  })(${args})`;
  const response = await sendCdp(tabId, "Runtime.evaluate", {
    expression,
    returnByValue: true,
  });
  const value = response.result?.value;
  if (!value?.ok) {
    throw new Error(`SelectOption 失败：${value?.error ?? "unknown"}`);
  }
  await dispatchKey(tabId, "Escape", 0);
  session.selectContexts.delete(tabId);
  await sleep(350);
  return { action: "select_option", selected: value };
}

async function configureWorkspace(session) {
  try {
    await refreshWorkspace(session);
  } catch (error) {
    session.workspace = disabledWorkspace(humanizeError(error));
  }
  await refreshFileTransferOnAttachedTabs(session);
  return publicWorkspace(session.workspace);
}

async function refreshWorkspace(session) {
  const handle = await loadWorkspaceHandle();
  if (!handle) throw new Error("请先在插件设置中选择文件 Workspace");
  if (!await verifyWorkspacePermission(handle, false)) {
    throw new Error("Workspace 目录已保存，但浏览器需要重新确认权限；请点击插件设置中的“恢复授权”");
  }
  session.workspace = {
    enabled: true,
    name: handle.name,
    files: await listWorkspaceFiles(handle),
    error: "",
  };
  return session.workspace;
}

async function refreshFileTransferOnAttachedTabs(session) {
  await Promise.allSettled(
    [...session.attachedTabs].map((tabId) => configureFileTransferForTab(session, tabId)),
  );
}

async function configureFileTransferForTab(session, tabId) {
  if (!session.workspace.enabled) {
    return;
  }
  await sendCdp(tabId, "Page.setInterceptFileChooserDialog", {
    enabled: true,
  });
  await configureDownloadGuard(tabId);
}

async function configureDownloadGuard(tabId) {
  try {
    await sendCdp(tabId, "Browser.setDownloadBehavior", {
      behavior: "deny",
      eventsEnabled: true,
    });
    return "browser";
  } catch (error) {
    if (!isMethodUnavailable(error)) throw error;
  }
  try {
    await sendCdp(tabId, "Page.setDownloadBehavior", { behavior: "deny" });
    return "page";
  } catch (error) {
    if (!isMethodUnavailable(error)) throw error;
    console.warn("Venus download guard is unavailable in this Chrome version");
    return "none";
  }
}

async function handleDebuggerEvent(source, method, params = {}) {
  const tabId = source?.tabId;
  if (method === "Page.fileChooserOpened" && Number.isInteger(tabId)) {
    const session = findSessionForTab(tabId);
    if (!session?.workspace.enabled) return;
    const promise = handleFileChooser(session, tabId, params);
    trackInteractionPromise(session, tabId, promise);
    await promise;
    return;
  }

  if (["Browser.downloadWillBegin", "Page.downloadWillBegin"].includes(method)) {
    const session = findSessionForTab(tabId);
    if (!session?.workspace.enabled) return;
    const url = await stableDownloadUrlForTab(tabId, params.url || "");
    queueRemoteDownload(session, tabId, {
      url,
      suggestedFilename: params.suggestedFilename || "",
    }, params.guid);
    return;
  }

  if (["Browser.downloadProgress", "Page.downloadProgress"].includes(method)) {
    const session = findSessionForDownload(params.guid) ?? findSessionForTab(tabId);
    const download = session?.downloads.get(params.guid);
    if (!session || !download || !["completed", "canceled"].includes(params.state)) return;
    if (download.kind === "remote") return;
    download.state = params.state;
    download.receivedBytes = params.receivedBytes;
    download.totalBytes = params.totalBytes;
    download.resolveCompletion?.(params.state);
    if (params.state === "canceled") {
      session.downloads.delete(params.guid);
    }
    recordFileTransfer(session, tabId, {
      type: "download",
      status: params.state === "completed" ? "ready_to_name" : "canceled",
      downloadId: params.guid,
      suggestedFilename: download.suggestedFilename,
      receivedBytes: params.receivedBytes,
      totalBytes: params.totalBytes,
    });
  }
}

async function handleFileChooser(session, tabId, params) {
  let event;
  try {
    await refreshWorkspace(session);
    const accept = await fileInputAccept(tabId, params.backendNodeId);
    session.pendingFileChoosers.set(tabId, {
      tabId,
      backendNodeId: params.backendNodeId,
      accept,
      mode: params.mode,
    });
    event = {
      type: "upload",
      status: "awaiting_selection",
      accept,
      mode: params.mode,
      files: publicWorkspaceFiles(listAcceptedWorkspaceFiles(session.workspace.files, accept)),
      instruction: "Choose one listed relative path with Upload(file='...').",
    };
  } catch (error) {
    event = {
      type: "upload",
      status: "failed",
      error: humanizeError(error),
      workspace: session.workspace.name,
    };
  }
  recordFileTransfer(session, tabId, event);
  return event;
}

async function uploadWorkspaceFile(session, requestedFile) {
  const chooser = latestPendingFileChooser(session);
  if (!chooser) {
    throw new Error("没有等待处理的文件选择器；请先点击上传控件");
  }
  await refreshWorkspace(session);
  const selected = findWorkspaceFile(session.workspace.files, requestedFile, chooser.accept);
  if (!selected) {
    throw new Error(`workspace 中不存在或不允许上传该文件：${requestedFile}`);
  }
  const directoryHandle = await requireWorkspaceHandle();
  const file = await getWorkspaceFile(directoryHandle, selected.relativePath);
  await injectFileInput(chooser.tabId, chooser.backendNodeId, file);
  session.pendingFileChoosers.delete(chooser.tabId);
  const event = {
    type: "upload",
    status: "selected",
    files: [selected.relativePath],
    workspace: session.workspace.name,
  };
  sendEvent(session, "file_transfer", event);
  await sleep(350);
  return { action: "upload", ...event };
}

async function finalizeDownload(session, rawFilename) {
  const filename = normalizeDownloadFilename(rawFilename);
  let download = [...session.downloads.values()]
    .filter((item) => !item.finalized)
    .sort((left, right) => left.startedAt - right.startedAt)[0];
  if (!download) {
    const tabId = await requireActiveTab(session);
    const tab = await chrome.tabs.get(tabId);
    const inferred = githubFileDownloadTarget(tab.url);
    if (inferred) {
      const guid = `github-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      download = {
        guid,
        kind: "remote",
        url: inferred.url,
        suggestedFilename: inferred.suggestedFilename,
        nativeDownloadIds: [],
        tabId,
        state: "ready",
        startedAt: Date.now(),
      };
      session.downloads.set(guid, download);
    }
  }
  if (!download) {
    throw new Error("没有等待命名的下载；请先点击下载控件");
  }
  if (download.kind === "remote") {
    const directoryHandle = await requireWorkspaceHandle();
    const tabId = session.attachedTabs.has(download.tabId)
      ? download.tabId
      : await requireActiveTab(session);
    await writeNetworkResourceToWorkspace(tabId, download.url, directoryHandle, filename);
    await cancelNativeDownloads(download);
    session.downloads.delete(download.guid);
    return completeDownloadAction(session, filename);
  }
  throw new Error("不支持的下载来源");
}

function completeDownloadAction(session, filename) {
  const event = {
    type: "download",
    status: "completed",
    filename,
    workspace: session.workspace.name,
  };
  sendEvent(session, "file_transfer", event);
  return { action: "download", ...event };
}

async function fileInputAccept(tabId, backendNodeId) {
  if (!Number.isInteger(backendNodeId)) return "";
  try {
    const response = await sendCdp(tabId, "DOM.describeNode", {
      backendNodeId,
      depth: 0,
    });
    const attributes = response.node?.attributes ?? [];
    for (let index = 0; index < attributes.length; index += 2) {
      if (String(attributes[index]).toLowerCase() === "accept") {
        return String(attributes[index + 1] ?? "");
      }
    }
  } catch {
    // The chooser node may disappear immediately after it opens.
  }
  return "";
}

function beginClickInteraction(session, tabId) {
  const interaction = { tabId, events: [], pending: new Set() };
  session.clickInteraction = interaction;
  return interaction;
}

function trackInteractionPromise(session, tabId, promise) {
  const interaction = session.clickInteraction;
  if (!interaction || interaction.tabId !== tabId) return;
  interaction.pending.add(promise);
  promise.finally(() => interaction.pending.delete(promise));
}

async function settleInteraction(interaction) {
  const pending = [...interaction.pending];
  if (pending.length) {
    await Promise.allSettled(pending);
  }
}

function recordFileTransfer(session, tabId, event) {
  const interaction = session.clickInteraction;
  if (interaction && (!Number.isInteger(tabId) || interaction.tabId === tabId)) {
    interaction.events.push(event);
  } else {
    session.pendingTransferNotices.push(event);
  }
  sendEvent(session, "file_transfer", event);
}

function findSessionForTab(tabId) {
  if (!Number.isInteger(tabId)) return null;
  return [...sessions.values()].find((session) => session.attachedTabs.has(tabId)) ?? null;
}

function findSessionForDownload(guid) {
  return [...sessions.values()].find((session) => session.downloads.has(guid)) ?? null;
}

async function handleNativeDownloadCreated(item) {
  if (!Number.isInteger(item?.id) || !Number.isInteger(item?.tabId)) return;
  const session = findSessionForTab(item.tabId);
  if (!session) return;

  const url = await stableDownloadUrlForTab(item.tabId, item.finalUrl || item.url || "");
  if (!url) return;
  queueRemoteDownload(session, item.tabId, {
    url,
    suggestedFilename: downloadBasename(item.filename),
    nativeDownloadId: item.id,
  });
  const queued = [...session.downloads.values()].find((download) =>
    download.nativeDownloadIds?.includes(item.id),
  );
  if (queued) {
    // The native download is only a signal used to discover the URL. Keep the
    // actual file out of Chrome's Downloads directory; Download(...) will
    // fetch it into workspace after the model chooses a filename.
    await cancelNativeDownloads(queued);
  }
}

async function stableDownloadUrlForTab(tabId, downloadUrl) {
  if (!String(downloadUrl).startsWith("blob:")) return downloadUrl;
  const tab = await chrome.tabs.get(tabId).catch(() => null);
  return stabilizeDownloadUrl(downloadUrl, tab?.url || "");
}

function queueRemoteDownload(session, tabId, target, preferredGuid = "") {
  const now = Date.now();
  const existing = [...session.downloads.values()].find((item) => (
    item.kind === "remote"
    && !item.finalized
    && item.tabId === tabId
    && now - item.startedAt < 10_000
    && (
      item.url === target.url
      || ((target.nativeDownloadId || preferredGuid) && now - item.startedAt < 3_000)
    )
  ));
  if (existing) {
    rememberNativeDownload(existing, target.nativeDownloadId);
    if (String(existing.url).startsWith("blob:") && /^https?:/i.test(target.url)) {
      existing.url = target.url;
    }
    if (!existing.suggestedFilename && target.suggestedFilename) {
      existing.suggestedFilename = target.suggestedFilename;
    }
    return publicDownloadTarget(existing, target);
  }

  const guid = preferredGuid || `remote-${now}-${Math.random().toString(36).slice(2)}`;
  const download = {
    guid,
    kind: "remote",
    url: target.url,
    suggestedFilename: target.suggestedFilename || "",
    nativeDownloadIds: [],
    tabId,
    state: "ready",
    startedAt: now,
  };
  rememberNativeDownload(download, target.nativeDownloadId);
  session.downloads.set(guid, download);
  recordFileTransfer(session, tabId, {
    type: "download",
    status: "awaiting_filename",
    downloadId: guid,
    url: target.url,
    linkText: target.text || "",
    suggestedFilename: download.suggestedFilename,
    instruction: "Choose the final name with Download(filename='...').",
  });
  return publicDownloadTarget(download, target);
}

function publicDownloadTarget(download, target = {}) {
  return {
    url: download.url,
    text: target.text || "",
    mimeType: target.mimeType || "",
    suggestedFilename: download.suggestedFilename || "",
    intercepted: true,
  };
}

function downloadBasename(path) {
  return String(path || "").replaceAll("\\", "/").split("/").at(-1) || "";
}

function rememberNativeDownload(download, downloadId) {
  if (!Number.isInteger(downloadId)) return;
  download.nativeDownloadIds ??= [];
  if (!download.nativeDownloadIds.includes(downloadId)) {
    download.nativeDownloadIds.push(downloadId);
  }
}

async function cancelNativeDownloads(download) {
  const ids = Array.isArray(download.nativeDownloadIds) ? download.nativeDownloadIds : [];
  await Promise.allSettled(ids.map(async (downloadId) => {
    await chrome.downloads.cancel(downloadId).catch(() => {});
    await chrome.downloads.erase({ id: downloadId }).catch(() => {});
  }));
}

function safeUrlProtocol(rawUrl) {
  try {
    return new URL(rawUrl).protocol;
  } catch {
    return "";
  }
}

function latestPendingFileChooser(session) {
  if (session.activeTabId && session.pendingFileChoosers.has(session.activeTabId)) {
    return session.pendingFileChoosers.get(session.activeTabId);
  }
  return [...session.pendingFileChoosers.values()].at(-1) ?? null;
}

function publicWorkspaceFiles(files) {
  return files.map((file) => ({
    path: file.relativePath,
    size: file.size,
    mimeType: file.mimeType,
  }));
}

async function requireWorkspaceHandle() {
  const handle = await loadWorkspaceHandle();
  if (!handle) throw new Error("请先在插件设置中选择文件 Workspace");
  if (!await verifyWorkspacePermission(handle, false)) {
    throw new Error("Workspace 目录已保存，但浏览器需要重新确认权限；请点击插件设置中的“恢复授权”");
  }
  return handle;
}

async function injectFileInput(tabId, backendNodeId, file) {
  const maxBytes = 64 * 1024 * 1024;
  if (file.size > maxBytes) {
    throw new Error("纯插件上传目前限制为 64 MiB");
  }
  const resolved = await sendCdp(tabId, "DOM.resolveNode", { backendNodeId });
  const objectId = resolved.object?.objectId;
  if (!objectId) throw new Error("上传控件已经失效，请重新点击");

  try {
    await callOnObject(tabId, objectId, "function() { this.__venusUploadChunks = []; }");
    const chunkSize = 768 * 1024;
    for (let offset = 0; offset < file.size; offset += chunkSize) {
      const bytes = new Uint8Array(await file.slice(offset, offset + chunkSize).arrayBuffer());
      await callOnObject(
        tabId,
        objectId,
        "function(chunk) { this.__venusUploadChunks.push(chunk); }",
        [{ value: bytesToBase64(bytes) }],
      );
    }
    await callOnObject(
      tabId,
      objectId,
      `function(name, type, lastModified) {
        const decoded = this.__venusUploadChunks.map((chunk) => atob(chunk));
        const length = decoded.reduce((total, chunk) => total + chunk.length, 0);
        const bytes = new Uint8Array(length);
        let offset = 0;
        for (const chunk of decoded) {
          for (let index = 0; index < chunk.length; index += 1) bytes[offset + index] = chunk.charCodeAt(index);
          offset += chunk.length;
        }
        const transfer = new DataTransfer();
        transfer.items.add(new File([bytes], name, { type, lastModified }));
        this.files = transfer.files;
        this.dispatchEvent(new Event('input', { bubbles: true }));
        this.dispatchEvent(new Event('change', { bubbles: true }));
        delete this.__venusUploadChunks;
      }`,
      [
        { value: file.name },
        { value: file.type || "application/octet-stream" },
        { value: file.lastModified },
      ],
    );
  } finally {
    await sendCdp(tabId, "Runtime.releaseObject", { objectId }).catch(() => {});
  }
}

async function callOnObject(tabId, objectId, functionDeclaration, args = []) {
  const result = await sendCdp(tabId, "Runtime.callFunctionOn", {
    objectId,
    functionDeclaration,
    arguments: args,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "页面文件注入失败");
  }
  return result.result?.value;
}

function bytesToBase64(bytes) {
  let binary = "";
  const blockSize = 32 * 1024;
  for (let offset = 0; offset < bytes.length; offset += blockSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + blockSize));
  }
  return btoa(binary);
}

async function writeNetworkResourceToWorkspace(tabId, url, directoryHandle, filename) {
  const resourceUrl = normalizeDownloadResourceUrl(url);
  if (["github.com", "raw.githubusercontent.com"].includes(new URL(resourceUrl).hostname)) {
    let extensionError;
    try {
      return await writeNetworkResourceViaExtension(resourceUrl, directoryHandle, filename);
    } catch (error) {
      extensionError = error;
      if (/缺少下载源访问权限/.test(humanizeError(error))) throw error;
    }
    try {
      return await writeNetworkResourceViaCdp(tabId, resourceUrl, directoryHandle, filename);
    } catch (cdpError) {
      throw new Error(
        `GitHub 下载失败：扩展请求 ${humanizeError(extensionError)}；浏览器请求 ${humanizeError(cdpError)}`,
      );
    }
  }
  await sendCdp(tabId, "Page.setBypassCSP", { enabled: true });
  try {
    let cdpError;
    try {
      // Prefer CDP's streaming response so large files are not buffered in the
      // page. CSP is temporarily bypassed because sites such as GitHub disallow
      // connecting to their raw-content host from the repository page.
      return await writeNetworkResourceViaCdp(tabId, resourceUrl, directoryHandle, filename);
    } catch (error) {
      cdpError = error;
      try {
        return await writeNetworkResourceViaPage(tabId, resourceUrl, directoryHandle, filename);
      } catch (pageError) {
        try {
          return await writeNetworkResourceViaExtension(resourceUrl, directoryHandle, filename);
        } catch (extensionError) {
          if (/缺少下载源访问权限/.test(humanizeError(extensionError))) throw extensionError;
          if (!isMethodUnavailable(cdpError) && !isMethodUnavailable(pageError)) throw cdpError;
          throw extensionError;
        }
      }
    }
  } finally {
    await sendCdp(tabId, "Page.setBypassCSP", { enabled: false }).catch(() => {});
  }
}

async function writeNetworkResourceViaExtension(url, directoryHandle, filename) {
  const maxBytes = 256 * 1024 * 1024;
  const parsedUrl = new URL(url);
  const origins = [`${parsedUrl.origin}/*`];
  if (!await chrome.permissions.contains({ origins })) {
    throw new Error(`插件缺少下载源访问权限：${parsedUrl.origin}`);
  }
  const response = await fetch(parsedUrl.href, {
    method: "GET",
    credentials: parsedUrl.hostname === "raw.githubusercontent.com" ? "omit" : "include",
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`下载服务器返回 HTTP ${response.status}`);
  }
  const contentLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(contentLength) && contentLength > maxBytes) {
    throw new Error("下载超过 256 MiB workspace 限制");
  }

  let writable;
  let fileCreated = false;
  try {
    const fileHandle = await createWorkspaceFile(directoryHandle, filename);
    fileCreated = true;
    writable = await fileHandle.createWritable();
    const reader = response.body?.getReader();
    if (!reader) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength > maxBytes) throw new Error("下载超过 256 MiB workspace 限制");
      await writable.write(bytes);
    } else {
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        received += value.byteLength;
        if (received > maxBytes) {
          await reader.cancel().catch(() => {});
          throw new Error("下载超过 256 MiB workspace 限制");
        }
        if (value.byteLength) await writable.write(value);
      }
    }
    await writable.close();
  } catch (error) {
    await writable?.abort().catch(() => {});
    if (fileCreated) await directoryHandle.removeEntry(filename).catch(() => {});
    throw error;
  }
}

async function writeNetworkResourceViaCdp(tabId, url, directoryHandle, filename) {
  const maxBytes = 256 * 1024 * 1024;
  const frameTree = await sendCdp(tabId, "Page.getFrameTree");
  const frameId = frameTree.frameTree?.frame?.id;
  if (!frameId) throw new Error("无法确定下载页面 frame");
  const loaded = await sendCdp(tabId, "Network.loadNetworkResource", {
    frameId,
    url,
    options: { disableCache: true, includeCredentials: true },
  });
  const resource = loaded.resource;
  if (!resource?.success || !resource.stream) {
    const reason = resource?.httpStatusCode
      ? `HTTP ${resource.httpStatusCode}`
      : resource?.netErrorName || "unknown error";
    throw new Error(`下载资源失败：${reason}`);
  }
  if (resource.httpStatusCode && (resource.httpStatusCode < 200 || resource.httpStatusCode >= 300)) {
    await sendCdp(tabId, "IO.close", { handle: resource.stream }).catch(() => {});
    throw new Error(`下载服务器返回 HTTP ${resource.httpStatusCode}`);
  }
  const contentLength = Number(headerValue(resource.headers, "content-length"));
  if (Number.isFinite(contentLength) && contentLength > maxBytes) {
    await sendCdp(tabId, "IO.close", { handle: resource.stream }).catch(() => {});
    throw new Error("下载超过 256 MiB workspace 限制");
  }
  let writable;
  let fileCreated = false;
  try {
    const fileHandle = await createWorkspaceFile(directoryHandle, filename);
    fileCreated = true;
    writable = await fileHandle.createWritable();
  } catch (error) {
    if (fileCreated) await directoryHandle.removeEntry(filename).catch(() => {});
    throw error;
  }
  let received = 0;
  try {
    while (true) {
      const chunk = await sendCdp(tabId, "IO.read", {
        handle: resource.stream,
        size: 1024 * 1024,
      });
      const bytes = chunk.base64Encoded
        ? base64ToBytes(chunk.data)
        : new TextEncoder().encode(chunk.data);
      received += bytes.byteLength;
      if (received > maxBytes) throw new Error("下载超过 256 MiB workspace 限制");
      if (bytes.byteLength) await writable.write(bytes);
      if (chunk.eof) break;
    }
    await writable.close();
  } catch (error) {
    await writable.abort().catch(() => {});
    await directoryHandle.removeEntry(filename).catch(() => {});
    throw error;
  } finally {
    await sendCdp(tabId, "IO.close", { handle: resource.stream }).catch(() => {});
  }
}

async function writeNetworkResourceViaPage(tabId, url, directoryHandle, filename) {
  const maxBytes = 256 * 1024 * 1024;
  const storageKey = `__venusDownload_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const expression = `(async () => {
    try {
      const response = await fetch(${JSON.stringify(url)}, {
        method: 'GET',
        credentials: 'same-origin',
        cache: 'no-store'
      });
      if (!response.ok) return { ok: false, error: 'HTTP ' + response.status };
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength > ${maxBytes}) return { ok: false, error: 'file_too_large' };
      globalThis[${JSON.stringify(storageKey)}] = bytes;
      return { ok: true, size: bytes.byteLength };
    } catch (error) {
      return { ok: false, error: String(error && error.message || error) };
    }
  })()`;
  const loaded = await sendCdp(tabId, "Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  const metadata = loaded.result?.value;
  if (!metadata?.ok) {
    const reason = metadata?.error === "file_too_large"
      ? "下载超过 256 MiB workspace 限制"
      : `页面下载失败：${metadata?.error ?? "unknown error"}`;
    throw new Error(reason);
  }

  let writable;
  let fileCreated = false;
  try {
    const fileHandle = await createWorkspaceFile(directoryHandle, filename);
    fileCreated = true;
    writable = await fileHandle.createWritable();
    const chunkSize = 768 * 1024;
    for (let offset = 0; offset < metadata.size; offset += chunkSize) {
      const chunkExpression = `(() => {
        const bytes = globalThis[${JSON.stringify(storageKey)}].subarray(${offset}, ${Math.min(offset + chunkSize, metadata.size)});
        let binary = '';
        for (let index = 0; index < bytes.length; index += 32768) {
          binary += String.fromCharCode(...bytes.subarray(index, index + 32768));
        }
        return btoa(binary);
      })()`;
      const chunk = await sendCdp(tabId, "Runtime.evaluate", {
        expression: chunkExpression,
        returnByValue: true,
      });
      await writable.write(base64ToBytes(chunk.result?.value ?? ""));
    }
    await writable.close();
  } catch (error) {
    await writable?.abort().catch(() => {});
    if (fileCreated) await directoryHandle.removeEntry(filename).catch(() => {});
    throw error;
  } finally {
    await sendCdp(tabId, "Runtime.evaluate", {
      expression: `delete globalThis[${JSON.stringify(storageKey)}]`,
    }).catch(() => {});
  }
}

function base64ToBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function headerValue(headers, wantedName) {
  const match = Object.entries(headers ?? {})
    .find(([name]) => name.toLowerCase() === wantedName.toLowerCase());
  return match?.[1] ?? "";
}

function disabledWorkspace(error = "") {
  return { enabled: false, name: "", files: [], error };
}

function publicWorkspace(workspace) {
  return {
    enabled: workspace.enabled,
    name: workspace.name,
    fileCount: workspace.files.length,
    error: workspace.error,
  };
}

async function adoptAgentOpenedTab(session) {
  if (!session.windowId) {
    return;
  }
  const tabs = await chrome.tabs.query({ windowId: session.windowId });
  const candidates = tabs.filter((tab) => {
    if (session.attachedTabs.has(tab.id) || !isControllableUrl(tab.url)) return false;
    if (session.ownedTabs.has(tab.id)) return true;
    return Number.isInteger(tab.openerTabId) && session.ownedTabs.has(tab.openerTabId);
  });
  if (!candidates.length) {
    return;
  }
  const target = candidates.find((tab) => tab.active) ?? candidates.at(-1);
  await attachTab(session, target.id);
  session.activeTabId = target.id;
  session.windowId = target.windowId;
  await bindPageAssistantToTab(session, target);
  session.viewport = null;
}

async function requireActiveTab(session) {
  if (session.activeTabId && session.attachedTabs.has(session.activeTabId)) {
    return session.activeTabId;
  }
  const fallback = [...session.attachedTabs].at(-1);
  if (!fallback) {
    throw new Error("当前没有已接管的标签页");
  }
  session.activeTabId = fallback;
  const tab = await chrome.tabs.get(fallback).catch(() => null);
  if (tab) await bindPageAssistantToTab(session, tab);
  return fallback;
}

function toCssPoint(session, point) {
  if (!Array.isArray(point) || point.length !== 2) {
    throw new Error("Action point 格式无效");
  }
  const [x, y] = point.map(Number);
  if (![x, y].every((value) => Number.isFinite(value) && value >= 0 && value <= 999)) {
    throw new Error("Action point 必须位于 0 到 999");
  }
  return {
    x: x / 999 * session.viewport.width,
    y: y / 999 * session.viewport.height,
  };
}

function detachAll(session) {
  if (session.detachPromise) return session.detachPromise;
  session.detachPromise = performDetachAll(session).finally(() => {
    session.detachPromise = null;
  });
  return session.detachPromise;
}

async function performDetachAll(session) {
  const tabIds = [...session.attachedTabs];
  if (session.workspace.enabled && tabIds.length) {
    await resetDownloadGuard(tabIds[0]);
  }
  session.attachedTabs.clear();
  session.ownedTabs.clear();
  session.selectContexts.clear();
  session.pendingFileChoosers.clear();
  session.activeTabId = null;
  session.bootstrapTabId = null;
  session.viewport = null;
  session.taskLabel = "";
  await Promise.allSettled(tabIds.map((tabId) => safeDetach(tabId)));
}

async function resetDownloadGuard(tabId) {
  try {
    await sendCdp(tabId, "Browser.setDownloadBehavior", {
      behavior: "default",
      eventsEnabled: false,
    });
  } catch (error) {
    if (!isMethodUnavailable(error)) return;
    await sendCdp(tabId, "Page.setDownloadBehavior", { behavior: "default" }).catch(() => {});
  }
}

async function safeDetach(tabId) {
  try {
    await chrome.debugger.detach({ tabId });
  } catch {
    // The tab may already be closed or detached by the user.
  }
}

function sendCdp(tabId, method, params = undefined) {
  return chrome.debugger.sendCommand({ tabId }, method, params);
}

function respond(session, requestId, ok, result = null, error = null) {
  if (!requestId || session.closed) {
    return;
  }
  try {
    session.port.postMessage({ requestId, ok, result, error });
  } catch {
    // The panel may have closed between command completion and response.
  }
}

function sendEvent(session, event, payload = {}) {
  if (session.closed) {
    return;
  }
  try {
    session.port.postMessage({ type: "event", event, payload });
  } catch {
    // Ignore closed ports.
  }
}

function tabSummary(tab) {
  return {
    id: tab.id,
    windowId: tab.windowId,
    title: tab.title ?? "",
    url: tab.url ?? "",
    status: tab.status ?? "",
  };
}

function isControllableUrl(rawUrl) {
  try {
    return ["http:", "https:"].includes(new URL(rawUrl).protocol);
  } catch {
    return false;
  }
}

function isBootstrapUrl(rawUrl) {
  const url = String(rawUrl || "").trim().toLowerCase();
  if (!url) return true;
  return url === "about:blank"
    || url.startsWith("chrome://newtab")
    || url.startsWith("chrome://new-tab-page")
    || url.startsWith("chrome-search://local-ntp");
}

async function waitForControllableTab(tabId, timeoutMs = 12_000) {
  const deadline = Date.now() + timeoutMs;
  let lastUrl = "";
  while (Date.now() < deadline) {
    let tab;
    try {
      tab = await chrome.tabs.get(tabId);
    } catch (error) {
      throw new Error(`目标标签页已关闭：${humanizeError(error)}`);
    }
    lastUrl = tab.url || tab.pendingUrl || lastUrl;
    if (isControllableUrl(tab.url)) return tab;
    await sleep(100);
  }
  throw new Error(`目标页面未进入可接管的 http/https 状态：${lastUrl || "unknown"}`);
}

function assertControllableUrl(rawUrl) {
  if (!isControllableUrl(rawUrl)) {
    throw new Error("Venus 目前只能接管 http/https 页面");
  }
}

function humanizeError(error) {
  return error instanceof Error ? error.message : String(error ?? "未知错误");
}

function isMethodUnavailable(error) {
  const message = humanizeError(error);
  return message.includes("-32601")
    || /(?:wasn't found|method not found)/i.test(message)
    // chrome.debugger can expose the Browser domain in the protocol schema
    // while still rejecting browser-target commands from a tab attachment.
    || /cannot\s+(?:not\s+)?access browser-level commands/i.test(message)
    // Some Chrome versions reject Network.loadNetworkResource when the
    // target page's CSP disallows the requested resource. The page-context
    // fetch fallback can still retrieve same-origin download URLs.
    || /csp violation/i.test(message);
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
