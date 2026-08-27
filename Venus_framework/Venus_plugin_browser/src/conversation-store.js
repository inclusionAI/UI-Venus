const DB_NAME = "venus-browser-agent";
const DB_VERSION = 1;
const CONVERSATIONS = "conversations";
const ENTRIES = "entries";
const ACTIVE_CONVERSATION_KEY = "activeConversationId";

export class ConversationStore {
  constructor({ indexedDBFactory = globalThis.indexedDB, storageArea = globalThis.chrome?.storage?.local } = {}) {
    if (!indexedDBFactory) {
      throw new Error("当前环境不支持 IndexedDB");
    }
    if (!storageArea) {
      throw new Error("当前环境不支持 chrome.storage.local");
    }
    this.indexedDB = indexedDBFactory;
    this.storage = storageArea;
    this.dbPromise = null;
  }

  async getOrCreateActiveConversation() {
    const saved = await this.storage.get([ACTIVE_CONVERSATION_KEY]);
    const savedId = saved[ACTIVE_CONVERSATION_KEY];
    if (savedId) {
      const conversation = await this.getConversation(savedId);
      if (conversation) return conversation;
    }

    const [latest] = await this.listConversations(1);
    if (latest) {
      await this.setActiveConversation(latest.id);
      return latest;
    }
    return this.createConversation();
  }

  async createConversation(title = "新会话") {
    const now = new Date().toISOString();
    const conversation = {
      id: makeId(),
      title: normalizeTitle(title),
      createdAt: now,
      updatedAt: now,
      nextSequence: 1,
      summary: "",
      summaryThrough: 0,
      promptTokens: 0,
      completionTokens: 0,
    };
    const db = await this.#db();
    const tx = db.transaction(CONVERSATIONS, "readwrite");
    tx.objectStore(CONVERSATIONS).add(conversation);
    await transactionDone(tx);
    await this.setActiveConversation(conversation.id);
    return conversation;
  }

  async setActiveConversation(conversationId) {
    const conversation = await this.getConversation(conversationId);
    if (!conversation) throw new Error("会话不存在");
    await this.storage.set({ [ACTIVE_CONVERSATION_KEY]: conversationId });
    return conversation;
  }

  async getConversation(conversationId) {
    if (!conversationId) return null;
    const db = await this.#db();
    const tx = db.transaction(CONVERSATIONS, "readonly");
    const result = await requestResult(tx.objectStore(CONVERSATIONS).get(conversationId));
    await transactionDone(tx);
    return result ?? null;
  }

  async listConversations(limit = 50) {
    const db = await this.#db();
    const tx = db.transaction(CONVERSATIONS, "readonly");
    const index = tx.objectStore(CONVERSATIONS).index("byUpdatedAt");
    const result = await collectCursor(index.openCursor(null, "prev"), limit);
    await transactionDone(tx);
    return result;
  }

  async renameConversation(conversationId, title) {
    return this.#updateConversation(conversationId, (conversation) => ({
      ...conversation,
      title: normalizeTitle(title),
      updatedAt: new Date().toISOString(),
    }));
  }

  async appendEntry(conversationId, entry) {
    const db = await this.#db();
    const tx = db.transaction([CONVERSATIONS, ENTRIES], "readwrite");
    const conversationStore = tx.objectStore(CONVERSATIONS);
    const conversation = await requestResult(conversationStore.get(conversationId));
    if (!conversation) {
      tx.abort();
      throw new Error("会话不存在，无法写入历史");
    }

    const sequence = conversation.nextSequence;
    const record = {
      ...cloneForStorage(entry),
      id: `${conversationId}:${sequence}`,
      conversationId,
      sequence,
      at: entry.at ?? new Date().toISOString(),
    };
    tx.objectStore(ENTRIES).add(record);
    conversationStore.put({
      ...conversation,
      nextSequence: sequence + 1,
      updatedAt: record.at,
    });
    await transactionDone(tx);
    return record;
  }

  async listEntries(conversationId, { afterSequence = 0 } = {}) {
    const db = await this.#db();
    const tx = db.transaction(ENTRIES, "readonly");
    const range = IDBKeyRange.bound(
      [conversationId, Math.max(0, afterSequence) + 1],
      [conversationId, Number.MAX_SAFE_INTEGER],
    );
    const entries = await requestResult(
      tx.objectStore(ENTRIES).index("byConversationSequence").getAll(range),
    );
    await transactionDone(tx);
    return entries ?? [];
  }

  async updateSummary(conversationId, summary, summaryThrough) {
    return this.#updateConversation(conversationId, (conversation) => ({
      ...conversation,
      summary: String(summary ?? "").trim(),
      summaryThrough: Math.max(conversation.summaryThrough ?? 0, Number(summaryThrough) || 0),
      promptTokens: 0,
      updatedAt: new Date().toISOString(),
    }));
  }

  async updatePromptTokens(conversationId, promptTokens) {
    const value = Number(promptTokens);
    if (!Number.isFinite(value) || value < 0) return this.getConversation(conversationId);
    return this.#updateConversation(conversationId, (conversation) => ({
      ...conversation,
      promptTokens: Math.round(value),
      updatedAt: new Date().toISOString(),
    }));
  }

  async addCompletionTokens(conversationId, completionTokens) {
    const increment = Number(completionTokens);
    if (!Number.isFinite(increment) || increment <= 0) return this.getConversation(conversationId);
    return this.#updateConversation(conversationId, (conversation) => ({
      ...conversation,
      completionTokens: Math.max(0, Number(conversation.completionTokens) || 0) + Math.round(increment),
      updatedAt: new Date().toISOString(),
    }));
  }

  async deleteConversation(conversationId) {
    const db = await this.#db();
    const tx = db.transaction([CONVERSATIONS, ENTRIES], "readwrite");
    const completion = transactionDone(tx);
    tx.objectStore(CONVERSATIONS).delete(conversationId);
    const index = tx.objectStore(ENTRIES).index("byConversationSequence");
    const range = IDBKeyRange.bound(
      [conversationId, 0],
      [conversationId, Number.MAX_SAFE_INTEGER],
    );
    await Promise.all([
      deleteCursor(index.openCursor(range)),
      completion,
    ]);

    const saved = await this.storage.get([ACTIVE_CONVERSATION_KEY]);
    if (saved[ACTIVE_CONVERSATION_KEY] === conversationId) {
      await this.storage.remove(ACTIVE_CONVERSATION_KEY);
    }
  }

  async #updateConversation(conversationId, transform) {
    const db = await this.#db();
    const tx = db.transaction(CONVERSATIONS, "readwrite");
    const store = tx.objectStore(CONVERSATIONS);
    const conversation = await requestResult(store.get(conversationId));
    if (!conversation) {
      tx.abort();
      throw new Error("会话不存在");
    }
    const updated = transform(conversation);
    store.put(updated);
    await transactionDone(tx);
    return updated;
  }

  #db() {
    if (!this.dbPromise) {
      this.dbPromise = openDatabase(this.indexedDB);
    }
    return this.dbPromise;
  }
}

function openDatabase(indexedDBFactory) {
  return new Promise((resolve, reject) => {
    const request = indexedDBFactory.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      const conversations = db.createObjectStore(CONVERSATIONS, { keyPath: "id" });
      conversations.createIndex("byUpdatedAt", "updatedAt");
      const entries = db.createObjectStore(ENTRIES, { keyPath: "id" });
      entries.createIndex(
        "byConversationSequence",
        ["conversationId", "sequence"],
        { unique: true },
      );
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("无法打开会话数据库"));
    request.onblocked = () => reject(new Error("会话数据库升级被其他插件页面阻塞"));
  });
}

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB 请求失败"));
  });
}

function transactionDone(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB 事务失败"));
    transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB 事务已取消"));
  });
}

function collectCursor(request, limit) {
  return new Promise((resolve, reject) => {
    const records = [];
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor || records.length >= limit) {
        resolve(records);
        return;
      }
      records.push(cursor.value);
      cursor.continue();
    };
    request.onerror = () => reject(request.error ?? new Error("读取会话列表失败"));
  });
}

function deleteCursor(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) {
        resolve();
        return;
      }
      const deletion = cursor.delete();
      deletion.onsuccess = () => cursor.continue();
      deletion.onerror = () => reject(deletion.error ?? new Error("删除会话历史失败"));
    };
    request.onerror = () => reject(request.error ?? new Error("删除会话历史失败"));
  });
}

function cloneForStorage(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function normalizeTitle(title) {
  const normalized = String(title ?? "").trim().replace(/\s+/g, " ");
  return (normalized || "新会话").slice(0, 72);
}

function makeId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
