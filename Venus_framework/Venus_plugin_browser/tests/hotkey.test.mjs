import test from "node:test";
import assert from "node:assert/strict";

import { editingCommandsForHotkey, normalizeHotkeyKeys } from "../src/hotkey.js";

test("maps platform-neutral and model-style ctrl shortcuts to Command on macOS", () => {
  assert.deepEqual(normalizeHotkeyKeys(["controlormeta", "a"], "mac"), ["meta", "a"]);
  assert.deepEqual(normalizeHotkeyKeys(["ctrl", "a"], "mac"), ["meta", "a"]);
  assert.deepEqual(normalizeHotkeyKeys(["cmd", "c"], "mac"), ["meta", "c"]);
});

test("keeps standard shortcuts on Control outside macOS", () => {
  assert.deepEqual(normalizeHotkeyKeys(["controlormeta", "a"], "win"), ["ctrl", "a"]);
  assert.deepEqual(normalizeHotkeyKeys(["ctrl", "a"], "linux"), ["ctrl", "a"]);
});

test("allows the physical Control and Option keys to be requested on macOS", () => {
  assert.deepEqual(normalizeHotkeyKeys(["control", "a"], "mac"), ["ctrl", "a"]);
  assert.deepEqual(normalizeHotkeyKeys(["option", "x"], "mac"), ["alt", "x"]);
});

test("adds an explicit selectAll editing command for the platform shortcut", () => {
  assert.deepEqual(editingCommandsForHotkey(["ctrl", "a"], "linux"), ["selectAll"]);
  assert.deepEqual(editingCommandsForHotkey(["ctrl", "a"], "win"), ["selectAll"]);
  assert.deepEqual(editingCommandsForHotkey(["meta", "a"], "mac"), ["selectAll"]);
});

test("does not reinterpret other shortcuts or physical Control+A on macOS", () => {
  assert.deepEqual(editingCommandsForHotkey(["ctrl", "a"], "mac"), []);
  assert.deepEqual(editingCommandsForHotkey(["ctrl", "shift", "a"], "linux"), []);
  assert.deepEqual(editingCommandsForHotkey(["ctrl", "c"], "linux"), []);
});
