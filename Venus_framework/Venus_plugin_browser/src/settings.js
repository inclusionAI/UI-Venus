const LOCAL_KEYS = ["apiUrl", "model", "rememberKey", "apiKey", "maxSteps", "temperature"];
const SESSION_KEYS = ["apiKey"];
export const DEFAULT_MAX_STEPS = 100;
export const DEFAULT_TEMPERATURE = 0.5;

export function normalizeApiEndpoint(input) {
  const raw = String(input ?? "").trim();
  if (!raw) {
    throw new Error("请填写 API URL");
  }
  const url = new URL(raw);
  const localHttp = url.protocol === "http:" && isLoopbackHost(url.hostname);
  if (url.protocol !== "https:" && !localHttp) {
    throw new Error("API URL 必须使用 HTTPS；只有 localhost/127.0.0.1 可以使用 HTTP");
  }
  url.hash = "";
  url.search = "";
  url.pathname = url.pathname.replace(/\/+$/, "");
  if (!url.pathname.endsWith("/chat/completions")) {
    url.pathname = `${url.pathname}/chat/completions`.replace(/\/{2,}/g, "/");
  }
  return url.href;
}

export function permissionPatternForApi(input) {
  const endpoint = new URL(normalizeApiEndpoint(input));
  const host = endpoint.hostname.includes(":") ? `[${endpoint.hostname}]` : endpoint.hostname;
  return `${endpoint.protocol}//${host}/*`;
}

export function normalizeMaxSteps(value = DEFAULT_MAX_STEPS) {
  const maxSteps = Number(value);
  if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 200) {
    throw new Error("最大步数必须是 1 到 200 之间的整数");
  }
  return maxSteps;
}

export function normalizeTemperature(value = DEFAULT_TEMPERATURE) {
  const temperature = Number(value);
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
    throw new Error("Temperature 必须是 0 到 2 之间的数字");
  }
  return temperature;
}

export async function requestApiPermission(apiUrl) {
  const origin = permissionPatternForApi(apiUrl);
  return chrome.permissions.request({ origins: [origin] });
}

export async function hasApiPermission(apiUrl) {
  const origin = permissionPatternForApi(apiUrl);
  return chrome.permissions.contains({ origins: [origin] });
}

export async function loadSettings() {
  const [local, session] = await Promise.all([
    chrome.storage.local.get(LOCAL_KEYS),
    chrome.storage.session.get(SESSION_KEYS),
  ]);
  return {
    apiUrl: local.apiUrl ?? "",
    model: local.model ?? "",
    rememberKey: Boolean(local.rememberKey),
    apiKey: session.apiKey ?? local.apiKey ?? "",
    maxSteps: safeSetting(normalizeMaxSteps, local.maxSteps, DEFAULT_MAX_STEPS),
    temperature: safeSetting(normalizeTemperature, local.temperature, DEFAULT_TEMPERATURE),
  };
}

export async function saveSettings(settings) {
  const apiUrl = normalizeApiEndpoint(settings.apiUrl);
  const model = String(settings.model ?? "").trim();
  const apiKey = String(settings.apiKey ?? "").trim();
  const rememberKey = Boolean(settings.rememberKey);
  const maxSteps = normalizeMaxSteps(settings.maxSteps);
  const temperature = normalizeTemperature(settings.temperature);

  if (!model) {
    throw new Error("请填写 Model");
  }
  if (!apiKey) {
    throw new Error("请填写 API Key");
  }

  await chrome.storage.local.setAccessLevel({ accessLevel: "TRUSTED_CONTEXTS" });
  await chrome.storage.session.setAccessLevel({ accessLevel: "TRUSTED_CONTEXTS" });
  await chrome.storage.local.set({ apiUrl, model, rememberKey, maxSteps, temperature });
  await chrome.storage.session.set({ apiKey });

  if (rememberKey) {
    await chrome.storage.local.set({ apiKey });
  } else {
    await chrome.storage.local.remove("apiKey");
  }

  return { apiUrl, model, apiKey, rememberKey, maxSteps, temperature };
}

function safeSetting(normalize, value, fallback) {
  try {
    return normalize(value ?? fallback);
  } catch {
    return fallback;
  }
}

function isLoopbackHost(hostname) {
  return ["localhost", "127.0.0.1", "::1", "[::1]"].includes(hostname.toLowerCase());
}
