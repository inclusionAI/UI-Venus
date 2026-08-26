export class BrowserBridge {
  constructor() {
    this.port = null;
    this.pending = new Map();
    this.sequence = 0;
    this.eventListeners = new Set();
  }

  connect() {
    if (this.port) {
      return;
    }
    this.port = chrome.runtime.connect({ name: "venus-agent-session" });
    this.port.onMessage.addListener((message) => this.#handleMessage(message));
    this.port.onDisconnect.addListener(() => this.#handleDisconnect());
  }

  onEvent(listener) {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  async attachCurrentTab(task = "") {
    return this.request("attach_current", { task });
  }

  async showControl() {
    return this.request("show_control");
  }

  async claimPageTask(tabId = null) {
    return this.request("claim_page_task", { tabId });
  }

  async setPanelVisibility(visible) {
    return this.request("set_panel_visibility", { visible }, 5_000);
  }

  async setControlEnabled(enabled) {
    return this.request("set_control_enabled", { enabled });
  }

  async configureWorkspace() {
    return this.request("configure_workspace", {}, 30_000);
  }

  async capture() {
    return this.request("capture", {}, 30_000);
  }

  async execute(action) {
    return this.request("execute", { action }, 180_000);
  }

  async updateControl(payload) {
    return this.request("update_control", { payload });
  }

  async detach() {
    if (!this.port) {
      return;
    }
    try {
      await this.request("detach", {}, 10_000);
    } catch (error) {
      // Closing the port triggers the service worker's independent cleanup
      // path when a detach request itself cannot complete or respond.
      const failedPort = this.port;
      this.port = null;
      failedPort?.disconnect();
      console.warn("Venus detach request failed; disconnected the session as fallback", error);
    }
  }

  async heartbeat() {
    if (!this.port) return;
    await this.request("control_heartbeat", {}, 5_000);
  }

  request(type, payload = {}, timeoutMs = 30_000) {
    this.connect();
    const requestId = `${Date.now()}-${++this.sequence}`;
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error(`浏览器命令超时：${type}`));
      }, timeoutMs);
      this.pending.set(requestId, { resolve, reject, timeoutId });
      this.port.postMessage({ requestId, type, ...payload });
    });
  }

  #handleMessage(message) {
    if (message?.type === "event") {
      for (const listener of this.eventListeners) {
        listener(message.event, message.payload ?? {});
      }
      return;
    }
    const pending = this.pending.get(message?.requestId);
    if (!pending) {
      return;
    }
    clearTimeout(pending.timeoutId);
    this.pending.delete(message.requestId);
    if (message.ok) {
      pending.resolve(message.result);
    } else {
      pending.reject(new Error(message.error ?? "浏览器命令失败"));
    }
  }

  #handleDisconnect() {
    const reason = chrome.runtime.lastError?.message ?? "插件后台连接已断开";
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeoutId);
      pending.reject(new Error(reason));
    }
    this.pending.clear();
    this.port = null;
    for (const listener of this.eventListeners) {
      listener("disconnect", { reason });
    }
  }
}
