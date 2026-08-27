import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const pageAssistantSource = await readFile(new URL("../page-assistant.js", import.meta.url), "utf8");
const serviceWorkerSource = await readFile(new URL("../service-worker.js", import.meta.url), "utf8");
const sidePanelSource = await readFile(new URL("../sidepanel.js", import.meta.url), "utf8");

test("uses a directly mounted textarea without an iframe initialization race", () => {
  assert.match(pageAssistantSource, /<textarea class="input"/);
  assert.doesNotMatch(pageAssistantSource, /<iframe|contentDocument/);
  assert.match(pageAssistantSource, /const acceptsTask = mode === "ready" \|\| mode === "complete"/);
});

test("is safe to reinject into a tab after an extension reload or message race", () => {
  assert.match(pageAssistantSource, /^\/\*\*[\s\S]*\(\(\) => \{/);
  assert.match(pageAssistantSource, /globalThis\[INSTANCE_KEY\]\?\.dispose\?\.\(\)/);
  assert.match(pageAssistantSource, /onMessage\.removeListener\(handleRuntimeMessage\)/);
  assert.match(pageAssistantSource, /\}\)\(\);\s*$/);
});

test("fails closed when the extension backend is unavailable", () => {
  assert.match(pageAssistantSource, /let enabled = false/);
  assert.match(pageAssistantSource, /enabled = response\?\.enabled === true/);
  assert.doesNotMatch(pageAssistantSource, /assistant-lease-expire/);
  assert.match(pageAssistantSource, /watchdogFailures \+= 1[\s\S]*watchdogFailures < 3[\s\S]*remove\(\)/);
});

test("separates task border visibility from side-panel assistant visibility", () => {
  assert.match(pageAssistantSource, /data-panel-visible/);
  assert.match(pageAssistantSource, /panelVisible = response\?\.panelVisible === true/);
  assert.match(serviceWorkerSource, /isTaskControlActive\(session\) \|\| shouldShowPanel/);
  assert.match(
    serviceWorkerSource,
    /enabled: Boolean\(visibleSession\) && \(isTaskControlActive\(visibleSession\) \|\| panelVisible\)/,
  );
});

test("renders transient notices as a top-centered toast", () => {
  assert.match(pageAssistantSource, /\.result \{ position: fixed; top: 24px; left: 50%; transform: translateX\(-50%\)/);
  assert.match(pageAssistantSource, /<\/div><aside class="result"/);
  assert.match(pageAssistantSource, /state\.noticeTimeoutMs > 0[\s\S]*view\.result\.hidden = true/);
  assert.match(pageAssistantSource, /else \{\s+view\.result\.hidden = true;/);
  assert.match(sidePanelSource, /restoreControlComposer\(text, 3_000\)/);
  assert.match(sidePanelSource, /mode: "complete"[\s\S]*noticeTimeoutMs: 3_000/);
  assert.match(sidePanelSource, /renderMessage\("system", "正在压缩上下文…"\)/);
});

test("removes the legacy debugger-injected control panel", () => {
  assert.match(pageAssistantSource, /LEGACY_HOST_ID = "__venus_control_indicator__"/);
  assert.match(pageAssistantSource, /document\.getElementById\(LEGACY_HOST_ID\)\?\.remove\(\)/);
  assert.match(
    serviceWorkerSource,
    /bindPageAssistantToTab[\s\S]*document\.getElementById\('__venus_control_indicator__'\)\?\.remove\(\)/,
  );
});

test("temporarily hides the assistant for screenshots without unmounting it", () => {
  assert.match(pageAssistantSource, /venus_page_assistant_capture_visibility/);
  assert.match(pageAssistantSource, /view\.host\.style\.visibility = message\.visible \? "visible" : "hidden"/);
  assert.match(serviceWorkerSource, /type: "venus_page_assistant_capture_visibility"/);
});

test("hides the page assistant when its side-panel session disconnects", () => {
  assert.match(serviceWorkerSource, /hiddenPageAssistantTabs\.add\(session\.assistantTabId\)/);
  assert.match(serviceWorkerSource, /setPageAssistantEnabled\(session\.assistantTabId, false\)/);
  assert.match(
    serviceWorkerSource,
    /stored\.controlPanelEnabled !== false\s+&& !hiddenPageAssistantTabs\.has\(tabId\)/,
  );
});

test("requires a visible side panel for the composer but not for a running-task border", () => {
  assert.match(
    serviceWorkerSource,
    /return isTaskControlActive\(session\) \|\| \(session\.panelVisible && session\.controlEnabled\)/,
  );
  assert.match(serviceWorkerSource, /panelVisible: false/);
});

test("publishes running state as soon as the task attaches", () => {
  assert.match(
    serviceWorkerSource,
    /session\.controlState = \{\s+mode: "running"[\s\S]*await bindPageAssistantToTab\(session, tab\);/,
  );
});

test("opens the side panel before the first async check on initial submission", () => {
  const handlerStart = serviceWorkerSource.indexOf("async function handlePageAssistantEvent");
  const handlerEnd = serviceWorkerSource.indexOf("async function pageTaskConfigurationError", handlerStart);
  const handlerSource = serviceWorkerSource.slice(handlerStart, handlerEnd);
  const queueIndex = handlerSource.indexOf("pendingPageTasks.set(tabId, task)");
  const openIndex = handlerSource.indexOf("chrome.sidePanel.open({ tabId })");
  const firstAwaitIndex = handlerSource.indexOf("await ");

  assert.ok(queueIndex >= 0 && queueIndex < openIndex);
  assert.ok(openIndex >= 0 && openIndex < firstAwaitIndex);
});

test("routes floating-panel tasks through the side panel task inbox", () => {
  assert.match(
    serviceWorkerSource,
    /pendingPageTasks\.set\(tabId, task\)[\s\S]*sendEvent\(session, "page_task_available", \{ tabId \}\)/,
  );
  assert.match(
    sidePanelSource,
    /elements\.taskInput\.value = pendingTask\.task;\s+await runTask\(\{ task: pendingTask\.task, source: "control" \}\)/,
  );
});

test("claims queued tasks by their original tab instead of the currently focused tab", () => {
  assert.match(sidePanelSource, /browser\.claimPageTask\(tabId\)/);
  assert.match(serviceWorkerSource, /case "claim_page_task":\s+result = await claimPendingPageTask\(session, message\.tabId\)/);
  assert.match(serviceWorkerSource, /chrome\.tabs\.get\(queuedTabId\)/);
});

test("mirrors side-panel document visibility to the page assistant", () => {
  assert.match(sidePanelSource, /if \(!HAS_NATIVE_SIDE_PANEL_VISIBILITY\)[\s\S]*document\.addEventListener\("visibilitychange", syncPanelVisibility\)/);
  assert.match(sidePanelSource, /if \(HAS_NATIVE_SIDE_PANEL_VISIBILITY\) return/);
  assert.match(sidePanelSource, /browser\.setPanelVisibility\(document\.visibilityState === "visible"\)/);
  assert.match(serviceWorkerSource, /case "set_panel_visibility":\s+result = await setPanelVisibility\(session, message\.visible\)/);
});

test("uses native side-panel open and close events when Chrome provides them", () => {
  assert.match(serviceWorkerSource, /chrome\.sidePanel\.onOpened\?\.addListener/);
  assert.match(serviceWorkerSource, /chrome\.sidePanel\.onClosed\?\.addListener/);
  assert.match(serviceWorkerSource, /handleNativeSidePanelVisibility\(info, true\)/);
  assert.match(serviceWorkerSource, /handleNativeSidePanelVisibility\(info, false\)/);
  assert.match(serviceWorkerSource, /activeTab\?\.id !== info\.tabId/);
  assert.match(
    serviceWorkerSource,
    /if \(Number\.isInteger\(info\?\.windowId\)\) return session\.windowId === info\.windowId/,
  );
});

test("moves the running page assistant when the agent changes tabs", () => {
  assert.match(
    serviceWorkerSource,
    /async function bindPageAssistantToTab\(session, tab\)[\s\S]*sendPageAssistantUpdate\(tab\.id, session\.controlState\)/,
  );
  assert.match(
    serviceWorkerSource,
    /async function launch[\s\S]*session\.activeTabId = tab\.id;[\s\S]*await bindPageAssistantToTab\(session, tab\)/,
  );
  assert.match(
    serviceWorkerSource,
    /async function adoptAgentOpenedTab[\s\S]*await bindPageAssistantToTab\(session, target\)/,
  );
});

test("moves an idle page assistant when the user activates another web tab", () => {
  assert.match(serviceWorkerSource, /chrome\.tabs\.onActivated\.addListener/);
  assert.match(serviceWorkerSource, /syncIdlePageAssistantTab\(tabId, windowId\)/);
  assert.match(
    serviceWorkerSource,
    /async function syncIdlePageAssistantTab[\s\S]*isControllableUrl\(tab\.url\)[\s\S]*bindPageAssistantToTab\(session, tab\)/,
  );
});

test("requests a workspace only when a download action needs it", () => {
  assert.doesNotMatch(sidePanelSource, /const workspace = await browser\.configureWorkspace\(\);\s+if \(workspace\.enabled\)/);
  assert.match(
    sidePanelSource,
    /if \(action\?\.name !== "download"\) return;\s+await ensureDownloadWorkspace\(signal\)/,
  );
  assert.match(sidePanelSource, /下载文件需要设置 Workspace 目录/);
  assert.match(serviceWorkerSource, /if \(isLikelyDownloadLink\(inspectedTarget\)\)/);
});

test("waits 500ms after every executed action before the next observation", () => {
  assert.match(
    serviceWorkerSource,
    /case "execute":\s+try \{\s+result = await executeAction\(session, message\.action\);\s+\} finally \{[\s\S]*await sleep\(500\);\s+\}\s+break;/,
  );
});
