import test from "node:test";
import assert from "node:assert/strict";

import { loadSettings, saveSettings } from "../src/settings.js";

test("starts with empty model credentials and remember-key disabled", async () => {
  globalThis.chrome = {
    storage: {
      local: new MemoryStorage(),
      session: new MemoryStorage(),
    },
  };

  const loaded = await loadSettings();
  assert.equal(loaded.apiUrl, "");
  assert.equal(loaded.model, "");
  assert.equal(loaded.apiKey, "");
  assert.equal(loaded.rememberKey, false);
  assert.equal(loaded.temperature, 0.5);
});

test("saves and reloads editable API URL, model and key", async () => {
  const local = new MemoryStorage();
  const session = new MemoryStorage();
  globalThis.chrome = { storage: { local, session } };

  await saveSettings({
    apiUrl: "https://api.example.com/v1",
    model: "custom-vision-model",
    apiKey: "custom-key",
    rememberKey: true,
    maxSteps: 42,
    temperature: 0.2,
  });
  const loaded = await loadSettings();

  assert.equal(loaded.apiUrl, "https://api.example.com/v1/chat/completions");
  assert.equal(loaded.model, "custom-vision-model");
  assert.equal(loaded.apiKey, "custom-key");
  assert.equal(loaded.rememberKey, true);
  assert.equal(loaded.maxSteps, 42);
  assert.equal(loaded.temperature, 0.2);
});

test("keeps a non-remembered key out of local storage", async () => {
  const local = new MemoryStorage();
  const session = new MemoryStorage();
  globalThis.chrome = { storage: { local, session } };

  await saveSettings({
    apiUrl: "https://api.example.com/v1",
    model: "custom-model",
    apiKey: "session-only-key",
    rememberKey: false,
    maxSteps: 100,
    temperature: 0.5,
  });

  assert.equal(local.data.apiKey, undefined);
  assert.equal(session.data.apiKey, "session-only-key");
  assert.equal((await loadSettings()).apiKey, "session-only-key");
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

  async setAccessLevel() {}
}
