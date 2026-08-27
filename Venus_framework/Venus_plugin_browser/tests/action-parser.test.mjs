import test from "node:test";
import assert from "node:assert/strict";

import {
  ActionParseError,
  parseAction,
  parseVenusResponse,
} from "../src/action-parser.js";
import {
  normalizeMaxSteps,
  normalizeApiEndpoint,
  normalizeTemperature,
  permissionPatternForApi,
} from "../src/settings.js";

test("parses all public point actions", () => {
  assert.deepEqual(parseAction("Click(point=(100,200))"), {
    name: "click",
    point: [100, 200],
    raw: "Click(point=(100,200))",
  });
  assert.equal(parseAction("DoubleClick(point=(1,2))").name, "double_click");
  assert.equal(parseAction("Hover(point=(3,4))").name, "hover");
  assert.deepEqual(parseAction("LongPress(point=(5,6), duration=2.5)"), {
    name: "long_press",
    point: [5, 6],
    duration: 2.5,
    raw: "LongPress(point=(5,6), duration=2.5)",
  });
  assert.deepEqual(parseAction("Drag(start=(10,20), end=(30,40))").end, [30, 40]);
  assert.deepEqual(parseAction("Scroll(direction='left', point=(50,60))"), {
    name: "scroll",
    point: [50, 60],
    direction: "left",
    raw: "Scroll(direction='left', point=(50,60))",
  });
});

test("parses text, navigation, hotkey and select actions", () => {
  assert.equal(parseAction("Type(content='hello, world')").content, "hello, world");
  assert.equal(
    parseAction("Finished(content='Here\'s one item, and another.')").content,
    "Here's one item, and another.",
  );
  assert.deepEqual(parseAction("Hotkey(keys=('ctrl','c'), repeat=2)").keys, ["ctrl", "c"]);
  assert.deepEqual(parseAction("SelectOption(index=3)").index, 3);
  assert.equal(parseAction("Launch(url='https://example.com/a')").url, "https://example.com/a");
  assert.equal(parseAction("PressBack()").name, "press_back");
  assert.equal(parseAction("GetUrl()").name, "get_url");
  assert.deepEqual(parseAction("Upload(file='inputs/report.pdf')"), {
    name: "upload",
    file: "inputs/report.pdf",
    raw: "Upload(file='inputs/report.pdf')",
  });
  assert.equal(parseAction("Download(filename='final-report.pdf')").filename, "final-report.pdf");
});

test("extracts one action from Venus response", () => {
  const result = parseVenusResponse(
    "<think>The search box is visible.</think><action>Click(point=(400,300))</action>",
  );
  assert.equal(result.think, "The search box is visible.");
  assert.equal(result.action.name, "click");
  assert.deepEqual(result.action.point, [400, 300]);
});

test("adds a missing opening think tag", () => {
  const result = parseVenusResponse(
    "The button is in the upper left.</think><action>Click(point=(10,50))</action>",
  );
  assert.equal(result.think, "The button is in the upper left.");
  assert.equal(
    result.rawResponse,
    "<think>The button is in the upper left.</think><action>Click(point=(10,50))</action>",
  );
  assert.deepEqual(result.action.point, [10, 50]);
});

test("rejects deleted actions, legacy box and unsafe URL schemes", () => {
  for (const action of [
    "TripleClick(point=(1,2))",
    "PressRecent()",
    "Click(box=(1,2))",
    "Launch(url='file:///tmp/a')",
    "Upload(file='../secret.txt')",
    "Download(filename='folder/report.pdf')",
  ]) {
    assert.throws(() => parseAction(action), ActionParseError);
  }
});

test("validates coordinates and long press duration", () => {
  assert.throws(() => parseAction("Click(point=(1000,1))"), /0 到 999/);
  assert.throws(() => parseAction("LongPress(point=(1,2), duration=60)"), /0 到 30/);
});

test("normalizes model endpoints and permission patterns", () => {
  assert.equal(
    normalizeApiEndpoint("https://api.example.com/v1/"),
    "https://api.example.com/v1/chat/completions",
  );
  assert.equal(
    normalizeApiEndpoint("http://127.0.0.1:8000/v1"),
    "http://127.0.0.1:8000/v1/chat/completions",
  );
  assert.equal(permissionPatternForApi("https://api.example.com/v1"), "https://api.example.com/*");
  assert.throws(() => normalizeApiEndpoint("http://api.example.com/v1"), /HTTPS/);
});

test("validates max steps and model temperature", () => {
  assert.equal(normalizeMaxSteps("50"), 50);
  assert.equal(normalizeTemperature("0.7"), 0.7);
  assert.throws(() => normalizeMaxSteps(0), /1 到 200/);
  assert.throws(() => normalizeMaxSteps(20.5), /整数/);
  assert.throws(() => normalizeTemperature(2.1), /0 到 2/);
});
