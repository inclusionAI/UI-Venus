export class ModelApiError extends Error {
  constructor(message, status = null, { retryable = false } = {}) {
    super(message);
    this.name = "ModelApiError";
    this.status = status;
    this.retryable = retryable;
  }
}

export class OpenAICompatibleClient {
  constructor({
    apiUrl,
    model,
    apiKey,
    temperature = 0,
    timeoutMs = 180_000,
    retryBaseDelayMs = 1_000,
    retryMaxDelayMs = 30_000,
    onRetry = () => {},
  }) {
    this.apiUrl = apiUrl;
    this.model = model;
    this.apiKey = apiKey;
    this.temperature = Number.isFinite(Number(temperature)) ? Number(temperature) : 0;
    this.timeoutMs = timeoutMs;
    this.retryBaseDelayMs = retryBaseDelayMs;
    this.retryMaxDelayMs = retryMaxDelayMs;
    this.onRetry = onRetry;
  }

  async complete(messages, signal = null, { onProgress = null } = {}) {
    const payload = {
      model: this.model,
      messages,
      temperature: this.temperature,
      max_tokens: 4096,
    };
    return this.#withInfiniteRetry(async () => {
      if (onProgress) {
        return this.#requestStream(payload, signal, onProgress);
      }
      const data = await this.#request(payload, signal);
      const message = data?.choices?.[0]?.message;
      if (!message) {
        throw new ModelApiError(
          "模型响应中缺少 choices[0].message",
          null,
          { retryable: true },
        );
      }
      const content = normalizeContent(message.content);
      if (!content.trim()) {
        throw new ModelApiError("模型返回了空 content", null, { retryable: true });
      }
      return {
        content,
        reasoningContent: String(message.reasoning_content ?? ""),
        usage: data.usage ?? null,
        raw: data,
      };
    }, signal);
  }

  async #requestStream(payload, externalSignal, onProgress) {
    if (externalSignal?.aborted) {
      throw new ModelApiError("模型请求已取消");
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(new Error("模型请求超时")), this.timeoutMs);
    const abortFromExternal = () => controller.abort(externalSignal?.reason);
    externalSignal?.addEventListener("abort", abortFromExternal, { once: true });

    try {
      const response = await fetch(this.apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          ...payload,
          stream: true,
          stream_options: { include_usage: true },
        }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const responseText = await response.text();
        let data = {};
        try {
          data = responseText ? JSON.parse(responseText) : {};
        } catch {
          // The status code still determines whether this request can be retried.
        }
        const detail = data?.error?.message ?? data?.message ?? responseText.slice(0, 500);
        throw new ModelApiError(
          `模型接口 ${response.status}：${detail}`,
          response.status,
          { retryable: isTransientStatus(response.status) },
        );
      }

      if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
        const data = await response.json();
        const message = data?.choices?.[0]?.message;
        const content = normalizeContent(message?.content);
        const reasoningContent = String(message?.reasoning_content ?? "");
        if (!content.trim()) {
          throw new ModelApiError("模型返回了空 content", null, { retryable: true });
        }
        await onProgress({ content, reasoningContent });
        return { content, reasoningContent, usage: data.usage ?? null, raw: data };
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let content = "";
      let reasoningContent = "";
      let usage = null;
      const chunks = [];
      const consumeEvent = async (eventText) => {
        const dataText = eventText
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n")
          .trim();
        if (!dataText || dataText === "[DONE]") return;
        let chunk;
        try {
          chunk = JSON.parse(dataText);
        } catch {
          throw new ModelApiError(
            `模型流返回了无效 JSON：${dataText.slice(0, 240)}`,
            null,
            { retryable: true },
          );
        }
        chunks.push(chunk);
        const delta = chunk?.choices?.[0]?.delta ?? {};
        content += normalizeContent(delta.content);
        reasoningContent += String(delta.reasoning_content ?? "");
        usage = chunk.usage ?? usage;
        await onProgress({ content, reasoningContent });
      };

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const events = buffer.split(/\r?\n\r?\n/);
        buffer = events.pop() ?? "";
        for (const eventText of events) await consumeEvent(eventText);
        if (done) break;
      }
      if (buffer.trim()) await consumeEvent(buffer);
      if (!content.trim()) {
        throw new ModelApiError("模型返回了空 content", null, { retryable: true });
      }
      return { content, reasoningContent, usage, raw: chunks };
    } catch (error) {
      if (controller.signal.aborted) {
        throw new ModelApiError(
          externalSignal?.aborted ? "模型请求已取消" : "模型请求超时",
          null,
          { retryable: !externalSignal?.aborted },
        );
      }
      if (error instanceof ModelApiError) throw error;
      throw new ModelApiError(
        `无法请求模型接口：${error.message}`,
        null,
        { retryable: true },
      );
    } finally {
      clearTimeout(timeoutId);
      externalSignal?.removeEventListener("abort", abortFromExternal);
    }
  }

  async test(signal = null) {
    const data = await this.#request({
      model: this.model,
      messages: [{ role: "user", content: "Reply with exactly OK." }],
      temperature: 0,
      max_tokens: 16,
    }, signal);
    const content = normalizeContent(data?.choices?.[0]?.message?.content);
    return content.trim();
  }

  async summarize(messages, signal = null) {
    const data = await this.#request({
      model: this.model,
      messages,
      temperature: 0,
      max_tokens: 1200,
    }, signal);
    const message = data?.choices?.[0]?.message;
    const content = normalizeContent(message?.content)
      || normalizeContent(message?.reasoning_content);
    if (!content.trim()) {
      throw new ModelApiError("模型没有返回上下文摘要");
    }
    return content.trim();
  }

  async #request(payload, externalSignal) {
    if (externalSignal?.aborted) {
      throw new ModelApiError("模型请求已取消");
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(new Error("模型请求超时")), this.timeoutMs);
    const abortFromExternal = () => controller.abort(externalSignal?.reason);
    externalSignal?.addEventListener("abort", abortFromExternal, { once: true });

    try {
      const response = await fetch(this.apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const responseText = await response.text();
      let data;
      try {
        data = responseText ? JSON.parse(responseText) : {};
      } catch {
        throw new ModelApiError(
          `模型接口返回了非 JSON 内容：${responseText.slice(0, 240)}`,
          response.status,
          { retryable: response.ok || isTransientStatus(response.status) },
        );
      }

      if (!response.ok) {
        const detail = data?.error?.message ?? data?.message ?? responseText.slice(0, 500);
        throw new ModelApiError(
          `模型接口 ${response.status}：${detail}`,
          response.status,
          { retryable: isTransientStatus(response.status) },
        );
      }
      return data;
    } catch (error) {
      if (controller.signal.aborted) {
        throw new ModelApiError(
          externalSignal?.aborted ? "模型请求已取消" : "模型请求超时",
          null,
          { retryable: !externalSignal?.aborted },
        );
      }
      if (error instanceof ModelApiError) {
        throw error;
      }
      throw new ModelApiError(
        `无法请求模型接口：${error.message}`,
        null,
        { retryable: true },
      );
    } finally {
      clearTimeout(timeoutId);
      externalSignal?.removeEventListener("abort", abortFromExternal);
    }
  }

  async #withInfiniteRetry(operation, signal) {
    let attempt = 0;
    while (true) {
      try {
        return await operation();
      } catch (error) {
        if (!(error instanceof ModelApiError) || !error.retryable || signal?.aborted) {
          throw error;
        }
        attempt += 1;
        const delayMs = Math.min(
          this.retryMaxDelayMs,
          this.retryBaseDelayMs * (2 ** Math.min(attempt - 1, 10)),
        );
        try {
          await this.onRetry({ attempt, delayMs, error });
        } catch (callbackError) {
          console.warn("Venus model retry callback failed", callbackError);
        }
        await abortableDelay(delayMs, signal);
      }
    }
  }
}

function isTransientStatus(status) {
  return [408, 409, 425, 429].includes(status) || status >= 500;
}

function abortableDelay(delayMs, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new ModelApiError("模型请求已取消"));
      return;
    }
    const timeoutId = setTimeout(() => {
      signal?.removeEventListener("abort", abortDelay);
      resolve();
    }, Math.max(0, delayMs));
    const abortDelay = () => {
      clearTimeout(timeoutId);
      reject(new ModelApiError("模型请求已取消"));
    };
    signal?.addEventListener("abort", abortDelay, { once: true });
  });
}

function normalizeContent(content) {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (typeof part === "string") {
        return part;
      }
      return typeof part?.text === "string" ? part.text : "";
    }).join("\n");
  }
  return "";
}
