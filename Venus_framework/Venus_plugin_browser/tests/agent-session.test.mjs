import test from "node:test";
import assert from "node:assert/strict";

import { AgentSession, AgentState } from "../src/agent-session.js";

test("injects persisted conversation context into a new task run", async () => {
  let receivedMessages = null;
  let detached = false;
  let detachCalls = 0;
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: "ZmFrZQ==",
      };
    },
    async execute() {
      throw new Error("Finished must not execute a browser action");
    },
    async detach() {
      detachCalls += 1;
      detached = true;
    },
  };
  const modelClient = {
    async complete(messages) {
      receivedMessages = messages;
      return {
        content: "<think>Done already.</think><action>Finished(content='done')</action>",
        usage: { prompt_tokens: 48_321, completion_tokens: 12 },
      };
    },
  };
  const states = [];
  let finalPayload = null;
  const session = new AgentSession({
    browser,
    modelClient,
    promptTemplate: "Task: {task} Date: {current_date}",
    conversationContext: "User task: remember the red product\nMessage (agent): saved",
    onState: ({ state }) => states.push(state),
    onFinal: (payload) => {
      finalPayload = payload;
      assert.equal(detached, true, "the browser must be released before completion is rendered");
    },
  });

  await session.run("continue");

  assert.match(receivedMessages[0].content, /Task: continue/);
  assert.match(receivedMessages[1].content[0].text, /remember the red product/);
  assert.equal(receivedMessages.at(-1).content.at(-1).type, "image_url");
  assert.equal(detached, true);
  assert.equal(detachCalls, 1);
  assert.equal(states.at(-1), AgentState.FINISHED);
  assert.equal(session.promptTokens, 48_321);
  assert.equal(session.contextTokens, 48_321);
  assert.equal(session.hasPromptTokenUsage, true);
  assert.equal(finalPayload.contextTokens, 48_321);
  assert.equal(finalPayload.outputTokens, 12);
});

test("releases the debugger before publishing a terminal result", async () => {
  let detached = false;
  let detachedBeforeFinal = false;
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: "shot",
      };
    },
    async execute() {},
    async detach() {
      detached = true;
    },
  };
  const session = new AgentSession({
    browser,
    modelClient: {
      async complete() {
        return { content: "<think>done</think><action>Finished(content='ready')</action>" };
      },
    },
    promptTemplate: "Task: {task}",
    onFinal: () => {
      detachedBeforeFinal = detached;
    },
  });

  await session.run("finish and release browser control");

  assert.equal(detachedBeforeFinal, true);
  assert.equal(detached, true);
  assert.equal(session.state, AgentState.FINISHED);
});

test("forwards the current step think while the model is streaming", async () => {
  const streamed = [];
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: "shot",
      };
    },
    async execute() {},
    async detach() {},
  };
  const session = new AgentSession({
    browser,
    modelClient: {
      async complete(_messages, _signal, options) {
        options.onProgress({ content: "<think>正在检查", reasoningContent: "" });
        options.onProgress({ content: "<think>正在检查页面</think>", reasoningContent: "" });
        return { content: "<think>正在检查页面</think><action>Finished(content='done')</action>" };
      },
    },
    promptTemplate: "Task: {task}",
    onThink: (value) => streamed.push(value),
  });

  await session.run("inspect page");

  assert.deepEqual(streamed, [
    { step: 1, think: "正在检查" },
    { step: 1, think: "正在检查页面" },
  ]);
});

test("uses reasoning_content as the persisted step analysis when content has no think block", async () => {
  let proposed = null;
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: "shot",
      };
    },
    async detach() {},
  };
  const session = new AgentSession({
    browser,
    modelClient: {
      async complete() {
        return {
          content: "<action>Finished(content='done')</action>",
          reasoningContent: "先确认任务已经完成，再结束操作。",
        };
      },
    },
    promptTemplate: "Task: {task}",
    onStep: (entry) => {
      if (entry.phase === "proposed") proposed = entry;
    },
  });

  await session.run("finish");

  assert.equal(proposed.think, "先确认任务已经完成，再结束操作。");
  assert.equal(
    proposed.rawResponse,
    "<think>先确认任务已经完成，再结束操作。</think>\n<action>Finished(content='done')</action>",
  );
});

test("sends canonical think and action assistant history on the next model request", async () => {
  const requests = [];
  let call = 0;
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: `shot-${call}`,
      };
    },
    async execute() {
      return { action: "wait" };
    },
    async detach() {},
  };
  const session = new AgentSession({
    browser,
    modelClient: {
      async complete(messages) {
        requests.push(messages);
        call += 1;
        return call === 1
          ? {
              content: "<action>Wait()</action>",
              reasoningContent: "等待页面稳定。",
            }
          : {
              content: "<think>页面已稳定。</think><action>Finished(content='done')</action>",
            };
      },
    },
    promptTemplate: "Task: {task}",
  });

  await session.run("wait then finish");

  const assistantHistory = requests[1].find((message) => (
    message.role === "assistant" && message.content.includes("Wait()")
  ));
  assert.equal(
    assistantHistory.content,
    "<think>等待页面稳定。</think>\n<action>Wait()</action>",
  );
});

test("sends at most 30 prior action rounds to the model", async () => {
  let calls = 0;
  let finalMessages = null;
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: `shot-${calls}`,
      };
    },
    async execute() {
      return { action: "wait" };
    },
    async detach() {},
  };
  const session = new AgentSession({
    browser,
    modelClient: {
      async complete(messages) {
        calls += 1;
        if (calls === 32) {
          finalMessages = messages;
          return { content: "<think>done</think><action>Finished(content='done')</action>" };
        }
        return { content: `<think>step ${calls}</think><action>Wait()</action>` };
      },
    },
    promptTemplate: "Task: {task}",
    maxSteps: 32,
  });

  await session.run("exercise the history window");

  const historyAssistants = finalMessages.filter((message) => message.role === "assistant");
  assert.equal(historyAssistants.length, 30);
  assert.doesNotMatch(historyAssistants[0].content, /step 1<\/think>/);
  assert.match(historyAssistants[0].content, /step 2<\/think>/);
  assert.match(historyAssistants.at(-1).content, /step 31<\/think>/);
});

test("sends user-attached images before the live browser screenshot", async () => {
  let receivedMessages;
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: "live-shot",
      };
    },
    async execute() {},
    async detach() {},
  };
  const modelClient = {
    async complete(messages) {
      receivedMessages = messages;
      return { content: "<think>done</think><action>Finished(content='done')</action>" };
    },
  };
  const session = new AgentSession({
    browser,
    modelClient,
    promptTemplate: "Task: {task}",
    taskImages: [{ name: "reference.png", dataUrl: "data:image/png;base64,dXNlcg==" }],
  });

  await session.run("use this image");

  const content = receivedMessages.at(-1).content;
  const label = content.find((part) => part.type === "text" && /reference/.test(part.text));
  assert.match(label.text, /reference\.png/);
  assert.deepEqual(
    content.filter((part) => part.type === "image_url").map((part) => part.image_url.url),
    ["data:image/png;base64,dXNlcg==", "data:image/jpeg;base64,live-shot"],
  );
});

test("sends the two preceding screenshots plus the current screenshot", async () => {
  let captureCount = 0;
  const requests = [];
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      captureCount += 1;
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: `shot-${captureCount}`,
      };
    },
    async execute() {
      return { action: "wait" };
    },
    async detach() {},
  };
  const modelClient = {
    async complete(messages) {
      requests.push(messages);
      const step = requests.length;
      return {
        content: step < 4
          ? `<think>wait ${step}</think><action>Wait()</action>`
          : "<think>done</think><action>Finished(content='done')</action>",
      };
    },
  };
  const session = new AgentSession({
    browser,
    modelClient,
    promptTemplate: "Task: {task}",
    maxSteps: 4,
  });

  await session.run("test image window");

  const imageUrls = requests.at(-1)
    .flatMap((message) => Array.isArray(message.content) ? message.content : [])
    .filter((part) => part.type === "image_url")
    .map((part) => part.image_url.url);
  assert.deepEqual(imageUrls, [
    "data:image/jpeg;base64,shot-2",
    "data:image/jpeg;base64,shot-3",
    "data:image/jpeg;base64,shot-4",
  ]);
});

test("adds a reflection warning after two identical consecutive responses", async () => {
  let modelCalls = 0;
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: "shot",
      };
    },
    async execute(action) {
      return { action: action.name };
    },
    async detach() {},
  };
  const repeatedResponse = "<think>the page has not changed</think><action>Wait()</action>";
  const modelClient = {
    async complete(messages) {
      modelCalls += 1;
      if (modelCalls <= 2) return { content: repeatedResponse };
      const currentUserText = messages.at(-1).content[0].text;
      assert.match(currentUserText, /⚠️ REFLECTION: You have output the exact same response for 2 consecutive steps\./);
      assert.match(currentUserText, /AVOID repeating the same action - try something different to make progress\./);
      return { content: "<think>change approach</think><action>Finished(content='done')</action>" };
    },
  };
  const session = new AgentSession({
    browser,
    modelClient,
    promptTemplate: "Task: {task}",
    maxSteps: 3,
  });

  await session.run("escape a loop");

  assert.equal(modelCalls, 3);
  assert.equal(session.state, AgentState.FINISHED);
});

test("returns action execution errors to the model and continues", async () => {
  let modelCalls = 0;
  let executeCalls = 0;
  let detached = false;
  const stepEvents = [];
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: `shot-${modelCalls + 1}`,
      };
    },
    async execute() {
      executeCalls += 1;
      throw new Error("click target vanished");
    },
    async detach() {
      detached = true;
    },
  };
  const modelClient = {
    async complete(messages) {
      modelCalls += 1;
      if (modelCalls === 1) {
        return {
          content: "<think>click it</think><action>Click(point=(500,500))</action>",
        };
      }
      const currentUserText = messages.at(-1).content[0].text;
      assert.match(currentUserText, /Result of the previous action/);
      assert.match(currentUserText, /click target vanished/);
      assert.match(currentUserText, /\"ok\": false/);
      return {
        content: "<think>stop after seeing the error</think><action>Finished(content='done')</action>",
      };
    },
  };
  const session = new AgentSession({
    browser,
    modelClient,
    promptTemplate: "Task: {task}",
    maxSteps: 3,
    onStep: (entry) => stepEvents.push(entry),
  });

  await session.run("recover from a failed click");

  assert.equal(modelCalls, 2);
  assert.equal(executeCalls, 1);
  assert.equal(detached, true);
  assert.equal(session.state, AgentState.FINISHED);
  const failed = stepEvents.find((entry) => entry.phase === "failed");
  assert.equal(failed.result.action, "click");
  assert.equal(failed.result.ok, false);
  assert.equal(failed.result.error, "click target vanished");
});

test("waits for beforeAction before executing the proposed action", async () => {
  let modelCalls = 0;
  let releaseAction;
  const order = [];
  const actionAllowed = new Promise((resolve) => {
    releaseAction = resolve;
  });
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Example", url: "https://example.com" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Example", url: "https://example.com" },
        viewport: { width: 1000, height: 700 },
        screenshot: "shot",
      };
    },
    async execute(action) {
      order.push(`execute:${action.name}`);
      return { action: action.name };
    },
    async detach() {},
  };
  const modelClient = {
    async complete() {
      modelCalls += 1;
      return modelCalls === 1
        ? { content: "<think>wait</think><action>Wait()</action>" }
        : { content: "<think>done</think><action>Finished(content='done')</action>" };
    },
  };
  const session = new AgentSession({
    browser,
    modelClient,
    promptTemplate: "Task: {task}",
    maxSteps: 2,
    beforeAction: async ({ action, observation }) => {
      order.push(`before:${action.name}:${observation.tab.url}`);
      await actionAllowed;
      order.push(`allowed:${action.name}`);
    },
  });

  const run = session.run("test action gate");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(order, ["before:wait:https://example.com"]);

  releaseAction();
  await run;

  assert.deepEqual(order, [
    "before:wait:https://example.com",
    "allowed:wait",
    "execute:wait",
  ]);
});

test("puts the complete upload workspace manifest in the next user message", async () => {
  let modelCalls = 0;
  const files = Array.from({ length: 300 }, (_, index) => ({
    path: `inputs/file-${index}.txt`,
    size: index,
    mimeType: "text/plain",
  }));
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Upload", url: "https://example.com/upload" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Upload", url: "https://example.com/upload" },
        viewport: { width: 1000, height: 700 },
        screenshot: "shot",
      };
    },
    async execute(action) {
      if (action.name === "click") {
        return {
          action: "click",
          fileTransfers: [{ type: "upload", status: "awaiting_selection", files }],
        };
      }
      return { action: "upload", status: "selected", files: [action.file] };
    },
    async detach() {},
  };
  const modelClient = {
    async complete(messages) {
      modelCalls += 1;
      if (modelCalls === 1) {
        return { content: "<think>open chooser</think><action>Click(point=(500,500))</action>" };
      }
      if (modelCalls === 2) {
        const nextUserMessage = messages.at(-1).content[0].text;
        assert.match(nextUserMessage, /You've clicked an upload button/);
        assert.match(nextUserMessage, /inputs\/file-299\.txt/);
        assert.match(nextUserMessage, /Upload\(file='exact\/relative\/path'\)/);
        assert.doesNotMatch(nextUserMessage, /"fileTransfers"/);
        assert.doesNotMatch(nextUserMessage, /Result of the previous action/);
        assert.doesNotMatch(nextUserMessage, /upload control accepts/i);
        assert.doesNotMatch(nextUserMessage, /\[truncated\]/);
        return { content: "<think>upload it</think><action>Upload(file='inputs/file-299.txt')</action>" };
      }
      return { content: "<think>done</think><action>Finished(content='done')</action>" };
    },
  };
  const session = new AgentSession({
    browser,
    modelClient,
    promptTemplate: "Task: {task}",
    maxSteps: 3,
  });

  await session.run("upload inputs/file-299.txt");
});

test("rejects Wait while a download is awaiting its final filename", async () => {
  let modelCalls = 0;
  const executed = [];
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Download", url: "https://github.com/example/repo/blob/main/README.md" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Download", url: "https://github.com/example/repo/blob/main/README.md" },
        viewport: { width: 1000, height: 700 },
        screenshot: "shot",
      };
    },
    async execute(action) {
      executed.push(action.name);
      if (action.name === "click") {
        return {
          action: "click",
          fileTransfers: [{
            type: "download",
            status: "awaiting_filename",
            suggestedFilename: "README.md",
          }],
        };
      }
      return { action: "download", status: "completed", filename: action.filename };
    },
    async detach() {},
  };
  const modelClient = {
    async complete(messages) {
      modelCalls += 1;
      if (modelCalls === 1) {
        return { content: "<think>click</think><action>Click(point=(900,200))</action>" };
      }
      if (modelCalls === 2) {
        return { content: "<think>wait</think><action>Wait()</action>" };
      }
      if (modelCalls === 3) {
        assert.match(messages.at(-1).content, /下一个 action 必须是 Download/);
        return { content: "<think>save</think><action>Download(filename='README.md')</action>" };
      }
      return { content: "<think>done</think><action>Finished(content='done')</action>" };
    },
  };
  const session = new AgentSession({
    browser,
    modelClient,
    promptTemplate: "Task: {task}",
    maxSteps: 3,
  });

  await session.run("download README.md");

  assert.deepEqual(executed, ["click", "download"]);
  assert.equal(modelCalls, 4);
});

test("rejects Finished when a download task has no completed download result", async () => {
  let modelCalls = 0;
  const executed = [];
  const browser = {
    async attachCurrentTab() {
      return { id: 1, title: "Download", url: "https://example.com/file" };
    },
    async capture() {
      return {
        tab: { id: 1, title: "Download", url: "https://example.com/file" },
        viewport: { width: 1000, height: 700 },
        screenshot: "shot",
      };
    },
    async execute(action) {
      executed.push(action.name);
      if (action.name === "click" && executed.length === 1) return { action: "click" };
      if (action.name === "click") {
        return {
          action: "click",
          fileTransfers: [{ type: "download", status: "awaiting_filename" }],
        };
      }
      return { action: "download", status: "completed", filename: action.filename };
    },
    async detach() {},
  };
  const modelClient = {
    async complete(messages) {
      modelCalls += 1;
      if (modelCalls === 1) {
        return { content: "<think>open file</think><action>Click(point=(500,500))</action>" };
      }
      if (modelCalls === 2) {
        return { content: "<think>done</think><action>Finished(content='downloaded')</action>" };
      }
      if (modelCalls === 3) {
        assert.match(messages.at(-1).content, /下载任务尚未完成/);
        return { content: "<think>click download</think><action>Click(point=(900,200))</action>" };
      }
      if (modelCalls === 4) {
        return { content: "<think>save</think><action>Download(filename='file.txt')</action>" };
      }
      return { content: "<think>done</think><action>Finished(content='downloaded')</action>" };
    },
  };
  const session = new AgentSession({
    browser,
    modelClient,
    promptTemplate: "Task: {task}",
    maxSteps: 4,
  });

  await session.run("下载这个文件");

  assert.deepEqual(executed, ["click", "click", "download"]);
  assert.equal(modelCalls, 5);
  assert.equal(session.state, AgentState.FINISHED);
});
