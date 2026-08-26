const DB_NAME = "venus-browser-workspace";
const DB_VERSION = 1;
const STORE_NAME = "handles";
const WORKSPACE_KEY = "workspace";

export async function saveWorkspaceHandle(handle, indexedDBFactory = globalThis.indexedDB) {
  if (!handle || handle.kind !== "directory") {
    throw new Error("请选择一个有效目录");
  }
  const db = await openDatabase(indexedDBFactory);
  const transaction = db.transaction(STORE_NAME, "readwrite");
  transaction.objectStore(STORE_NAME).put(handle, WORKSPACE_KEY);
  await transactionDone(transaction);
  db.close();
}

export async function loadWorkspaceHandle(indexedDBFactory = globalThis.indexedDB) {
  const db = await openDatabase(indexedDBFactory);
  const transaction = db.transaction(STORE_NAME, "readonly");
  const handle = await requestResult(transaction.objectStore(STORE_NAME).get(WORKSPACE_KEY));
  await transactionDone(transaction);
  db.close();
  return handle ?? null;
}

export async function verifyWorkspacePermission(handle, request = false) {
  if (!handle) return false;
  const options = { mode: "readwrite" };
  if (await handle.queryPermission(options) === "granted") return true;
  return request && await handle.requestPermission(options) === "granted";
}

export async function listWorkspaceFiles(directoryHandle, prefix = "") {
  const files = [];
  for await (const entry of directoryHandle.values()) {
    if (entry.name.startsWith(".")) continue;
    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.kind === "directory") {
      files.push(...await listWorkspaceFiles(entry, relativePath));
      continue;
    }
    const file = await entry.getFile();
    files.push({
      relativePath,
      name: file.name,
      size: file.size,
      mimeType: file.type || "application/octet-stream",
      modifiedAt: new Date(file.lastModified).toISOString(),
    });
  }
  return files.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
}

export async function getWorkspaceFile(directoryHandle, relativePath) {
  const parts = safeRelativeParts(relativePath);
  let directory = directoryHandle;
  for (const part of parts.slice(0, -1)) {
    directory = await directory.getDirectoryHandle(part);
  }
  const handle = await directory.getFileHandle(parts.at(-1));
  return handle.getFile();
}

export async function createWorkspaceFile(directoryHandle, filename) {
  if (!filename || filename === "." || filename === ".." || /[\\/\0]/.test(filename)) {
    throw new Error("下载文件名必须是单个安全文件名");
  }
  try {
    await directoryHandle.getFileHandle(filename);
    throw new Error(`workspace 已存在同名文件：${filename}`);
  } catch (error) {
    if (error?.name !== "NotFoundError") throw error;
  }
  return directoryHandle.getFileHandle(filename, { create: true });
}

function safeRelativeParts(value) {
  const path = String(value ?? "").trim().replaceAll("\\", "/").replace(/^\.\//, "");
  const parts = path.split("/");
  if (!path || path.startsWith("/") || parts.some((part) => !part || part === "." || part === "..")) {
    throw new Error("workspace 文件必须是安全的相对路径");
  }
  return parts;
}

function openDatabase(indexedDBFactory) {
  if (!indexedDBFactory) throw new Error("当前环境不支持 IndexedDB");
  return new Promise((resolve, reject) => {
    const request = indexedDBFactory.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("无法打开 workspace 存储"));
  });
}

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("workspace 存储请求失败"));
  });
}

function transactionDone(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error ?? new Error("workspace 存储事务已取消"));
    transaction.onerror = () => reject(transaction.error ?? new Error("workspace 存储事务失败"));
  });
}
