import test from "node:test";
import assert from "node:assert/strict";

import { OpenAICompatibleClient } from "../src/model-client.js";

test("uses configured temperature for action completions", async () => {
  const originalFetch = globalThis.fetch;
  let payload = null;
  globalThis.fetch = async (_url, options) => {
    payload = JSON.parse(options.body);
    return new Response(JSON.stringify({
      choices: [{ message: { content: "<think>ok</think><action>Wait()</action>" } }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const client = new OpenAICompatibleClient({
      apiUrl: "https://api.example.com/v1/chat/completions",
      model: "vision-model",
      apiKey: "test",
      temperature: 0.7,
    });
    await client.complete([{ role: "user", content: "test" }]);
    assert.equal(payload.temperature, 0.7);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("uses the default temperature for action completions", async () => {
  const originalFetch = globalThis.fetch;
  let payload = null;
  globalThis.fetch = async (_url, options) => {
    payload = JSON.parse(options.body);
    return new Response(JSON.stringify({
      choices: [{ message: { content: "<think>ok</think><action>Wait()</action>" } }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const client = new OpenAICompatibleClient({
      apiUrl: "https://api.example.com/v1/chat/completions",
      model: "vision-model",
      apiKey: "test",
    });
    await client.complete([{ role: "user", content: "test" }]);
    assert.equal(payload.temperature, 0.5);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("treats a successful model response without content as a valid connection", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({
    choices: [{ message: { content: "", reasoning_content: "internal reasoning" } }],
  }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

  try {
    const client = new OpenAICompatibleClient({
      apiUrl: "https://api.example.com/v1/chat/completions",
      model: "vision-model",
      apiKey: "test",
    });
    assert.equal(await client.test(), "");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("streams cumulative model output through onProgress", async () => {
  const originalFetch = globalThis.fetch;
  let payload = null;
  globalThis.fetch = async (_url, options) => {
    payload = JSON.parse(options.body);
    return new Response([
      'data: {"choices":[{"delta":{"content":"<think>正在"}}]}',
      "",
      'data: {"choices":[{"delta":{"content":"检查</think>"}}]}',
      "",
      'data: {"choices":[{"delta":{"content":"<action>Wait()</action>"}}]}',
      "",
      'data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":17,"total_tokens":137}}',
      "",
      "data: [DONE]",
      "",
    ].join("\n"), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  };

  try {
    const progress = [];
    const client = new OpenAICompatibleClient({
      apiUrl: "https://api.example.com/v1/chat/completions",
      model: "vision-model",
      apiKey: "test",
    });
    const response = await client.complete(
      [{ role: "user", content: "test" }],
      null,
      { onProgress: (value) => progress.push(value) },
    );
    assert.equal(payload.stream, true);
    assert.deepEqual(payload.stream_options, { include_usage: true });
    assert.equal(response.content, "<think>正在检查</think><action>Wait()</action>");
    assert.equal(progress.at(-1).content, response.content);
    assert.equal(response.usage.completion_tokens, 17);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("retries transient gateway failures until an action response succeeds", async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  const retries = [];
  globalThis.fetch = async () => {
    calls += 1;
    if (calls <= 2) {
      return new Response("<html><h1>504 Gateway Time-out</h1></html>", {
        status: 504,
        headers: { "Content-Type": "text/html" },
      });
    }
    return new Response(JSON.stringify({
      choices: [{ message: { content: "<think>recovered</think><action>Wait()</action>" } }],
    }), { status: 200 });
  };

  try {
    const client = new OpenAICompatibleClient({
      apiUrl: "https://api.example.com/v1/chat/completions",
      model: "vision-model",
      apiKey: "test",
      retryBaseDelayMs: 0,
      retryMaxDelayMs: 0,
      onRetry: (event) => retries.push(event),
    });
    const response = await client.complete([{ role: "user", content: "test" }]);
    assert.match(response.content, /recovered/);
    assert.equal(calls, 3);
    assert.deepEqual(retries.map((event) => event.attempt), [1, 2]);
    assert.equal(retries[0].error.status, 504);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("does not retry permanent authorization errors", async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return new Response("<html><h1>403 Forbidden</h1></html>", { status: 403 });
  };

  try {
    const client = new OpenAICompatibleClient({
      apiUrl: "https://api.example.com/v1/chat/completions",
      model: "vision-model",
      apiKey: "bad-key",
      retryBaseDelayMs: 0,
    });
    await assert.rejects(
      client.complete([{ role: "user", content: "test" }]),
      (error) => error.status === 403 && error.retryable === false,
    );
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("stopping a task aborts the retry wait", async () => {
  const originalFetch = globalThis.fetch;
  const controller = new AbortController();
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return new Response("gateway unavailable", { status: 503 });
  };

  try {
    const client = new OpenAICompatibleClient({
      apiUrl: "https://api.example.com/v1/chat/completions",
      model: "vision-model",
      apiKey: "test",
      retryBaseDelayMs: 60_000,
      onRetry: () => controller.abort(),
    });
    await assert.rejects(
      client.complete([{ role: "user", content: "test" }], controller.signal),
      /已取消/,
    );
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
