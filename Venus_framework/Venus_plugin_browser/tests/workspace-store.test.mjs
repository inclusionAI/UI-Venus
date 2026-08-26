import test from "node:test";
import assert from "node:assert/strict";

import {
  createWorkspaceFile,
  getWorkspaceFile,
  listWorkspaceFiles,
  verifyWorkspacePermission,
} from "../src/workspace-store.js";

class FakeFileHandle {
  constructor(name, content, type = "text/plain") {
    this.kind = "file";
    this.name = name;
    this.content = content;
    this.type = type;
  }

  async getFile() {
    return {
      name: this.name,
      size: this.content.length,
      type: this.type,
      lastModified: 1_700_000_000_000,
      text: async () => this.content,
    };
  }
}

class FakeDirectoryHandle {
  constructor(name, entries = []) {
    this.kind = "directory";
    this.name = name;
    this.entries = new Map(entries.map((entry) => [entry.name, entry]));
  }

  async *values() {
    yield* this.entries.values();
  }

  async getDirectoryHandle(name) {
    const entry = this.entries.get(name);
    if (entry?.kind !== "directory") throw notFound();
    return entry;
  }

  async getFileHandle(name, { create = false } = {}) {
    const entry = this.entries.get(name);
    if (entry?.kind === "file") return entry;
    if (!create) throw notFound();
    const created = new FakeFileHandle(name, "");
    this.entries.set(name, created);
    return created;
  }
}

test("recursively lists visible workspace files", async () => {
  const nested = new FakeDirectoryHandle("documents", [
    new FakeFileHandle("paper.pdf", "pdf", "application/pdf"),
  ]);
  const root = new FakeDirectoryHandle("workspace", [
    new FakeFileHandle("notes.txt", "notes"),
    new FakeFileHandle(".hidden", "secret"),
    nested,
  ]);

  const files = await listWorkspaceFiles(root);

  assert.deepEqual(files.map((file) => file.relativePath), ["documents/paper.pdf", "notes.txt"]);
  assert.equal(await (await getWorkspaceFile(root, "documents/paper.pdf")).text(), "pdf");
});

test("creates downloads without overwriting existing files", async () => {
  const root = new FakeDirectoryHandle("workspace", [new FakeFileHandle("existing.txt", "old")]);
  assert.equal((await createWorkspaceFile(root, "new.txt")).name, "new.txt");
  await assert.rejects(() => createWorkspaceFile(root, "existing.txt"), /同名/);
  await assert.rejects(() => createWorkspaceFile(root, "../escape.txt"), /安全/);
});

test("checks persisted directory permissions", async () => {
  const handle = {
    async queryPermission() { return "prompt"; },
    async requestPermission() { return "granted"; },
  };
  assert.equal(await verifyWorkspacePermission(handle, false), false);
  assert.equal(await verifyWorkspacePermission(handle, true), true);
});

function notFound() {
  const error = new Error("not found");
  error.name = "NotFoundError";
  return error;
}
