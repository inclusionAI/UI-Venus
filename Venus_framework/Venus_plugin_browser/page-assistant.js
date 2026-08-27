/**
 * 使用示例：开启 Side Panel 顶部的“页面助手”后，可在网页底部输入任务并按 Ctrl/⌘ + Enter 运行。
 */
(() => {
const INSTANCE_KEY = "__venusPageAssistantInstance__";
try {
  globalThis[INSTANCE_KEY]?.dispose?.();
} catch {
  // The previous content-script context may have been invalidated by an
  // extension reload. Its DOM is still safe to replace below.
}

const HOST_ID = "__venus_page_assistant__";
const LEGACY_HOST_ID = "__venus_control_indicator__";
const DEFAULT_STATE = {
  mode: "ready",
  task: "Venus Browser Agent",
  think: "输入新任务",
};

let enabled = false;
let panelVisible = false;
let state = DEFAULT_STATE;
let view = null;
let leaseTimer = null;
let noticeTimer = null;
let watchdogFailures = 0;

globalThis[INSTANCE_KEY] = { dispose };
initialize();

async function initialize() {
  document.getElementById(LEGACY_HOST_ID)?.remove();
  const response = await chrome.runtime.sendMessage({ type: "venus_page_assistant_ready" }).catch(() => null);
  enabled = response?.enabled === true;
  panelVisible = response?.panelVisible === true;
  if (enabled) mount();
  if (response?.state) applyState(response.state);
  if (enabled) renewLease();
}

chrome.runtime.onMessage.addListener(handleRuntimeMessage);

function handleRuntimeMessage(message) {
  if (message?.type === "venus_page_assistant_update") {
    applyState(message.state);
  } else if (message?.type === "venus_page_assistant_visible") {
    enabled = Boolean(message.visible);
    panelVisible = Boolean(message.panelVisible);
    if (enabled) {
      mount();
      applyPanelVisibility();
      applyState(state);
      renewLease();
    } else {
      remove();
    }
  } else if (message?.type === "venus_page_assistant_capture_visibility") {
    if (view) view.host.style.visibility = message.visible ? "visible" : "hidden";
  } else if (message?.type === "venus_page_assistant_interactive") {
    if (view) view.host.dataset.interactive = message.interactive ? "true" : "false";
  }
}

function dispose() {
  try {
    chrome.runtime.onMessage.removeListener(handleRuntimeMessage);
  } catch {
    // Expected when this instance belongs to an invalidated extension context.
  }
  enabled = false;
  remove();
}

function mount() {
  if (!enabled || view || !document.documentElement) return;
  document.getElementById(HOST_ID)?.remove();
  const host = document.createElement("div");
  host.id = HOST_ID;
  host.dataset.mode = "ready";
  host.dataset.interactive = "false";
  host.dataset.panelVisible = panelVisible ? "true" : "false";
  host.style.cssText = "all:initial;position:fixed;inset:0;z-index:2147483647;pointer-events:none;display:block;";
  const shadow = host.attachShadow({ mode: "closed" });
  shadow.innerHTML = `
    <style>
      :host { all: initial; }
      .layer { position: fixed; inset: 0; pointer-events: none; overflow: hidden; }
      .grid { position: absolute; inset: 0; background-image: radial-gradient(circle, rgb(111 145 255 / 34%) 1px, transparent 1.25px); background-size: 11px 11px; opacity: .19; animation: grid-breathe 2.8s ease-in-out infinite; }
      .frame { position: absolute; inset: 0; border: 3px solid rgb(103 139 255 / 62%); border-radius: 8px; box-shadow: inset 0 0 24px rgb(88 119 255 / 34%), inset 0 0 70px rgb(112 76 220 / 16%); animation: frame-pulse 1.8s ease-in-out infinite; }
      .shell { position: absolute; left: 50%; bottom: 20px; transform: translateX(-50%); display: flex; align-items: flex-end; gap: 10px; width: max-content; max-width: calc(100vw - 32px); }
      .panel { width: min(440px, calc(100vw - 32px)); box-sizing: border-box; padding: 9px 12px; border: 1px solid rgb(255 255 255 / 17%); border-radius: 18px; color: #fff; background: rgb(24 27 39 / 88%); box-shadow: 0 12px 38px rgb(20 24 48 / 36%), inset 0 1px rgb(255 255 255 / 10%); backdrop-filter: blur(13px); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
      .status-row { display: flex; align-items: center; gap: 12px; min-height: 36px; }
      .stop { all: initial; position: relative; display: block; width: 30px; height: 30px; flex: 0 0 30px; border-radius: 50%; cursor: pointer; pointer-events: auto; background: rgb(145 169 255 / 10%); transition: border-radius .25s ease, background .25s ease, transform .25s ease; }
      .stop::before, .stop::after { content: ''; position: absolute; inset: 6px; border: 2px solid #bca8ff; border-radius: 50%; transition: inset .25s ease, border-radius .25s ease; }
      .stop::after { inset: 11px; border-color: #fff; }
      .stop:hover, .stop:focus-visible { border-radius: 8px; background: rgb(145 169 255 / 24%); transform: scale(1.08); }
      .stop:hover::before, .stop:focus-visible::before { inset: 5px; border-radius: 4px; }
      .stop:hover::after, .stop:focus-visible::after { inset: 10px; border-radius: 2px; }
      :host([data-interactive='false']) .stop { pointer-events: none; }
      :host(:not([data-mode='running'])) .stop { display: none; }
      .copy { min-width: 0; }
      .task { overflow: hidden; color: #fff; font-size: 13px; font-weight: 700; line-height: 1.35; text-overflow: ellipsis; white-space: nowrap; }
      .state { display: flex; align-items: center; gap: 6px; margin-top: 3px; color: #c9d3ff; font-size: 11px; font-weight: 600; }
      .dot { width: 6px; height: 6px; flex: 0 0 6px; border-radius: 50%; background: #91a9ff; box-shadow: 0 0 9px #91a9ff; animation: dot 1.2s ease-in-out infinite; }
      .state-label { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .composer { display: flex; align-items: flex-end; gap: 8px; margin-top: 8px; }
      .input { all: initial; display: block; width: 100%; min-width: 0; height: 36px; box-sizing: border-box; resize: none; overflow: hidden; padding: 8px 11px; border: 1px solid rgb(173 188 255 / 28%); border-radius: 12px; outline: 0; color: #fff; background: rgb(255 255 255 / 8%); caret-color: #fff; font: 13px/18px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; pointer-events: auto; transition: border-color .18s ease, box-shadow .18s ease; }
      .input::placeholder { color: rgb(215 221 255 / 58%); }
      .input:focus { border-color: rgb(162 179 255 / 68%); box-shadow: 0 0 0 2px rgb(116 139 255 / 18%); }
      .input:disabled, .submit:disabled { opacity: .55; cursor: default; }
      .submit { all: initial; display: grid; place-items: center; width: 36px; height: 36px; flex: 0 0 36px; border-radius: 50%; color: #fff; background: linear-gradient(145deg, #8e7cff, #668cff); box-shadow: 0 5px 16px rgb(92 112 255 / 38%); cursor: pointer; pointer-events: auto; transition: transform .18s ease, filter .18s ease, box-shadow .18s ease; }
      .submit svg { width: 18px; height: 18px; pointer-events: none; transition: transform .18s ease; }
      .submit:hover, .submit:focus-visible { transform: translateY(-1px) scale(1.06); filter: brightness(1.08); box-shadow: 0 8px 22px rgb(92 112 255 / 48%); }
      .submit:hover svg, .submit:focus-visible svg { transform: translateX(2px); }
      .submit:active { transform: scale(.94); }
      .result { position: fixed; top: 24px; left: 50%; transform: translateX(-50%); width: min(420px, calc(100vw - 32px)); max-height: 180px; box-sizing: border-box; overflow: auto; padding: 12px 34px 12px 14px; border: 1px solid rgb(180 192 255 / 28%); border-radius: 15px; color: #eef1ff; background: rgb(31 34 48 / 94%); box-shadow: 0 12px 36px rgb(20 24 48 / 34%); backdrop-filter: blur(13px); pointer-events: auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
      .result-title { color: #b9c6ff; font-size: 11px; font-weight: 750; }
      .result-content { margin: 5px 0 0; color: #fff; font-size: 13px; line-height: 1.45; white-space: pre-wrap; }
      .close { all: initial; position: absolute; top: 7px; right: 9px; display: grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; color: #cbd3f5; cursor: pointer; pointer-events: auto; font: 18px/1 sans-serif; transition: color .18s ease, background .18s ease, transform .18s ease; }
      .close:hover, .close:focus-visible { color: #fff; background: rgb(255 255 255 / 10%); transform: scale(1.08); }
      [hidden] { display: none !important; }
      :host([data-panel-visible='false']) .panel, :host([data-panel-visible='false']) .result { display: none !important; }
      :host(:not([data-mode='running'])) .grid, :host(:not([data-mode='running'])) .frame { display: none; }
      @keyframes grid-breathe { 50% { opacity: .1; } }
      @keyframes frame-pulse { 50% { border-color: rgb(150 119 255 / 78%); box-shadow: inset 0 0 34px rgb(100 135 255 / 42%); } }
      @keyframes dot { 50% { opacity: .35; transform: scale(.72); } }
      @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
    </style>
    <div class="layer"><div class="grid"></div><div class="frame"></div><div class="shell">
      <div class="panel"><div class="status-row"><button class="stop" type="button" aria-label="停止 Venus Agent"></button><span class="copy"><span class="task"></span><span class="state"><span class="dot"></span><span class="state-label"></span></span></span></div>
      <form class="composer" hidden><textarea class="input" rows="1" placeholder="输入新任务…" aria-label="任务"></textarea><button class="submit" type="submit" aria-label="运行任务"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3.5 10h12M11 5.5l4.5 4.5-4.5 4.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></button></form></div>
    </div><aside class="result" aria-live="polite" hidden><button class="close" type="button" aria-label="关闭结果">×</button><div class="result-title"></div><p class="result-content"></p></aside></div>`;

  document.documentElement.append(host);
  view = {
    host,
    task: shadow.querySelector(".task"),
    state: shadow.querySelector(".state-label"),
    composer: shadow.querySelector(".composer"),
    input: shadow.querySelector(".input"),
    submit: shadow.querySelector(".submit"),
    stop: shadow.querySelector(".stop"),
    result: shadow.querySelector(".result"),
    resultTitle: shadow.querySelector(".result-title"),
    resultContent: shadow.querySelector(".result-content"),
  };
  for (const eventName of ["keydown", "keyup", "keypress", "beforeinput", "input", "change", "compositionstart", "compositionupdate", "compositionend", "paste", "copy", "cut", "click", "dblclick", "mousedown", "mouseup", "pointerdown", "pointerup", "touchstart", "touchend"]) {
    shadow.addEventListener(eventName, (event) => event.stopPropagation());
  }
  view.stop.addEventListener("click", () => {
    view.stop.disabled = true;
    view.host.dataset.interactive = "false";
    view.state.textContent = "正在停止…";
    sendEvent("stop");
  });
  view.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    const task = view.input.value.trim();
    if (!task || view.submit.disabled) return view.input.focus();
    applyState({ mode: "starting", task, think: "正在启动…" });
    sendEvent("run_task", { task });
  });
  view.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      view.composer.requestSubmit();
    }
  });
  shadow.querySelector(".close").addEventListener("click", () => { view.result.hidden = true; });
  applyState(state);
  renewLease();
}

function remove() {
  clearTimeout(leaseTimer);
  clearTimeout(noticeTimer);
  leaseTimer = null;
  noticeTimer = null;
  document.getElementById(HOST_ID)?.remove();
  view = null;
}

function renewLease() {
  if (!enabled || !view) return;
  clearTimeout(leaseTimer);
  leaseTimer = setTimeout(async () => {
    const response = await chrome.runtime.sendMessage({ type: "venus_page_assistant_ready" }).catch(() => null);
    if (!response) {
      watchdogFailures += 1;
      if (watchdogFailures < 3) {
        renewLease();
        return;
      }
      enabled = false;
      remove();
      return;
    }
    watchdogFailures = 0;
    enabled = response.enabled === true;
    panelVisible = response.panelVisible === true;
    if (!enabled) {
      remove();
      return;
    }
    applyPanelVisibility();
    if (response.state) applyState(response.state);
    renewLease();
  }, 4_000);
}

function applyPanelVisibility() {
  if (view) view.host.dataset.panelVisible = panelVisible ? "true" : "false";
}

function applyState(next = {}) {
  clearTimeout(noticeTimer);
  noticeTimer = null;
  const previousMode = state.mode;
  state = { ...state, ...next };
  if (!enabled) return;
  if (!view) mount();
  if (!view) return;
  const mode = ["running", "starting", "complete", "ready"].includes(state.mode) ? state.mode : "ready";
  view.host.dataset.mode = mode;
  view.host.dataset.interactive = mode === "running" ? "true" : "false";
  view.task.textContent = state.task || "Venus Browser Agent";
  view.state.textContent = state.think || (mode === "starting" ? "正在启动…" : "就绪");
  view.composer.hidden = mode === "running" || mode === "starting";
  view.stop.disabled = mode !== "running";
  const acceptsTask = mode === "ready" || mode === "complete";
  view.input.disabled = !acceptsTask;
  view.submit.disabled = !acceptsTask;
  if (mode === "running" && previousMode === "starting") view.input.value = "";
  if (mode === "complete" && state.content) {
    view.result.hidden = false;
    view.resultTitle.textContent = state.outcome === "call_user" ? "需要用户处理" : "任务完成";
    view.resultContent.textContent = state.content;
    if (state.noticeTimeoutMs > 0) {
      noticeTimer = setTimeout(() => {
        if (view) view.result.hidden = true;
      }, state.noticeTimeoutMs);
    }
  } else if (mode === "ready" && state.notice) {
    view.result.hidden = false;
    view.resultTitle.textContent = "无法运行任务";
    view.resultContent.textContent = state.notice;
    if (state.noticeTimeoutMs > 0) {
      noticeTimer = setTimeout(() => {
        if (view) view.result.hidden = true;
      }, state.noticeTimeoutMs);
    }
  } else {
    view.result.hidden = true;
  }
}

function sendEvent(event, payload = {}) {
  chrome.runtime.sendMessage({ type: "venus_page_assistant_event", event, ...payload }).catch(() => {});
}
})();
