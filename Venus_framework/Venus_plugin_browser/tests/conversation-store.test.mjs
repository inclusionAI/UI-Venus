import test from "node:test";
import assert from "node:assert/strict";

import { IDBFactory, IDBKeyRange } from "fake-indexeddb";

import { ConversationStore } from "../src/conversation-store.js";

globalThis.IDBKeyRange = IDBKeyRange;

test("deletes a conversation together with its existing entries", async () => {
  const storage = new MemoryStorage();
  const store = new ConversationStore({
    indexedDBFactory: new IDBFactory(),
    storageArea: storage,
  });
  const preserved = await store.createConversation("保留会话");
  const deleted = await store.createConversation("待删除会话");
  await store.appendEntry(deleted.id, { kind: "message", role: "user", text: "已有内容" });
  await store.appendEntry(deleted.id, { kind: "result", step: 1, result: { action: "click" } });

  await store.deleteConversation(deleted.id);

  assert.equal(await store.getConversation(deleted.id), null);
  assert.deepEqual(await store.listEntries(deleted.id), []);
  assert.equal((await store.getConversation(preserved.id)).title, "保留会话");
  assert.equal(storage.data.activeConversationId, undefined);
});

test("persists actual prompt usage and clears it after compaction", async () => {
  const store = new ConversationStore({
    indexedDBFactory: new IDBFactory(),
    storageArea: new MemoryStorage(),
  });
  const conversation = await store.createConversation("长会话");

  await store.updatePromptTokens(conversation.id, 51_234);
  assert.equal((await store.getConversation(conversation.id)).promptTokens, 51_234);

  await store.updateSummary(conversation.id, "用户让我查询商品，结果已给出。", 10);
  assert.equal((await store.getConversation(conversation.id)).promptTokens, 0);
});

test("accumulates completion tokens across tasks in one conversation", async () => {
  const store = new ConversationStore({
    indexedDBFactory: new IDBFactory(),
    storageArea: new MemoryStorage(),
  });
  const conversation = await store.createConversation("多任务会话");

  await store.addCompletionTokens(conversation.id, 120);
  await store.addCompletionTokens(conversation.id, 80);

  assert.equal((await store.getConversation(conversation.id)).completionTokens, 200);
});

class MemoryStorage {
  constructor() {
    this.data = {};
  }

  async get(keys) {
    return Object.fromEntries(
      keys
        .filter((key) => Object.hasOwn(this.data, key))
        .map((key) => [key, this.data[key]]),
    );
  }

  async set(values) {
    Object.assign(this.data, values);
  }

  async remove(key) {
    delete this.data[key];
  }
}
