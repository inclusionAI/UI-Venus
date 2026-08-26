export class ActionParseError extends Error {
  constructor(message, rawAction = "") {
    super(message);
    this.name = "ActionParseError";
    this.rawAction = rawAction;
  }
}

const POINT_ACTIONS = new Map([
  ["click", "click"],
  ["doubleclick", "double_click"],
  ["hover", "hover"],
]);

const NO_ARG_ACTIONS = new Map([
  ["wait", "wait"],
  ["geturl", "get_url"],
  ["pressback", "press_back"],
  ["presshome", "press_home"],
  ["pressenter", "press_enter"],
]);

const TEXT_ACTIONS = new Map([
  ["type", "type"],
  ["takenote", "take_note"],
  ["calluser", "call_user"],
  ["finished", "finished"],
]);

function normalizeResponseContent(content) {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((part) => part && (part.type === "text" || typeof part.text === "string"))
      .map((part) => String(part.text ?? ""))
      .join("\n");
  }
  return "";
}

export function parseVenusResponse(content) {
  const rawText = normalizeResponseContent(content).trim();
  if (!rawText) {
    throw new ActionParseError("模型返回了空内容");
  }
  // Keep parity with the Python Venus agent: some reasoning models emit the
  // closing </think> token but omit the opening token from message.content.
  const text = rawText.startsWith("<think>") ? rawText : `<think>${rawText}`;

  const thinkMatch = text.match(/<think>\s*([\s\S]*?)\s*<\/think>/i);
  const actionMatch = text.match(/<action>\s*([\s\S]*?)\s*<\/action>/i);
  let rawAction = actionMatch?.[1]?.trim();

  if (!rawAction) {
    const fallback = text.match(/(?:^|\n)\s*Action\s*:\s*([^\n]+)\s*$/im);
    rawAction = fallback?.[1]?.trim();
  }
  if (!rawAction) {
    throw new ActionParseError("模型输出中缺少 <action>...</action>");
  }

  return {
    think: thinkMatch?.[1]?.trim() ?? "",
    rawResponse: text,
    rawAction,
    action: parseAction(rawAction),
  };
}

export function parseAction(rawAction) {
  const raw = String(rawAction ?? "").trim();
  const match = raw.match(/^([A-Za-z_]\w*)\s*\(([\s\S]*)\)$/);
  if (!match) {
    throw new ActionParseError("Action 必须是函数调用格式", raw);
  }

  const originalName = match[1];
  const name = originalName.toLowerCase().replaceAll("_", "");
  const rawParams = match[2].trim();

  if (name === "tripleclick" || name === "pressrecent") {
    throw new ActionParseError(`${originalName} 已从 action space 删除`, raw);
  }

  if (NO_ARG_ACTIONS.has(name)) {
    if (rawParams) {
      throw new ActionParseError(`${originalName} 不接受参数`, raw);
    }
    return { name: NO_ARG_ACTIONS.get(name), raw };
  }

  if (TEXT_ACTIONS.has(name)) {
    const content = parseSingleTextParameter(rawParams, "content", name === "finished");
    return { name: TEXT_ACTIONS.get(name), content, raw };
  }

  const params = parseParameters(rawParams);

  if (name === "upload") {
    const file = unquote(required(params, "file", originalName)).trim();
    if (!file || file.startsWith("/") || file.replaceAll("\\", "/").split("/").some((part) => !part || part === "." || part === "..")) {
      throw new ActionParseError("Upload file 必须是 workspace 中的安全相对路径", raw);
    }
    return { name: "upload", file, raw };
  }

  if (name === "download") {
    const filename = unquote(required(params, "filename", originalName)).trim();
    if (!filename || filename === "." || filename === ".." || /[\\/\0]/.test(filename)) {
      throw new ActionParseError("Download filename 必须是单个安全文件名", raw);
    }
    return { name: "download", filename, raw };
  }

  if (POINT_ACTIONS.has(name)) {
    return {
      name: POINT_ACTIONS.get(name),
      point: parsePoint(required(params, "point", originalName)),
      raw,
    };
  }

  if (name === "longpress") {
    const duration = params.has("duration") ? parseNumber(params.get("duration")) : 20;
    if (!Number.isFinite(duration) || duration <= 0 || duration > 30) {
      throw new ActionParseError("LongPress duration 必须在 0 到 30 秒之间", raw);
    }
    return {
      name: "long_press",
      point: parsePoint(required(params, "point", originalName)),
      duration,
      raw,
    };
  }

  if (name === "drag") {
    return {
      name: "drag",
      start: parsePoint(required(params, "start", originalName)),
      end: parsePoint(required(params, "end", originalName)),
      raw,
    };
  }

  if (name === "scroll") {
    const direction = unquote(params.get("direction") ?? "down").toLowerCase();
    if (!["up", "down", "left", "right"].includes(direction)) {
      throw new ActionParseError(`不支持的滚动方向：${direction}`, raw);
    }
    return {
      name: "scroll",
      point: parsePoint(required(params, "point", originalName)),
      direction,
      raw,
    };
  }

  if (name === "launch") {
    const url = unquote(required(params, "url", originalName));
    let parsed;
    try {
      parsed = new URL(url);
    } catch {
      throw new ActionParseError(`Launch URL 无效：${url}`, raw);
    }
    if (!["http:", "https:"].includes(parsed.protocol)) {
      throw new ActionParseError("Launch 仅允许 http/https URL", raw);
    }
    return { name: "launch", url: parsed.href, raw };
  }

  if (name === "hotkey") {
    const keys = parseKeys(required(params, "keys", originalName));
    const repeat = params.has("repeat") ? parseInteger(params.get("repeat")) : 1;
    if (keys.length === 0 || keys.length > 3) {
      throw new ActionParseError("Hotkey 必须包含 1 到 3 个按键", raw);
    }
    if (!Number.isInteger(repeat) || repeat < 1 || repeat > 20) {
      throw new ActionParseError("Hotkey repeat 必须在 1 到 20 之间", raw);
    }
    return { name: "hotkey", keys, repeat, raw };
  }

  if (name === "selectoption") {
    const action = { name: "select_option", raw };
    if (params.has("index")) {
      action.index = parseInteger(params.get("index"));
    }
    if (params.has("value")) {
      action.value = unquote(params.get("value"));
    }
    if (params.has("text")) {
      action.text = unquote(params.get("text"));
    }
    if (action.index === undefined && action.value === undefined && action.text === undefined) {
      throw new ActionParseError("SelectOption 需要 index、value 或 text", raw);
    }
    return action;
  }

  throw new ActionParseError(`未知 action：${originalName}`, raw);
}

function required(params, key, actionName) {
  if (!params.has(key)) {
    throw new ActionParseError(`${actionName} 缺少 ${key} 参数`);
  }
  return params.get(key);
}

function parseParameters(rawParams) {
  const params = new Map();
  if (!rawParams) {
    return params;
  }
  for (const part of splitTopLevel(rawParams)) {
    const separator = findTopLevelEquals(part);
    if (separator < 1) {
      throw new ActionParseError(`无法解析参数：${part}`);
    }
    const key = part.slice(0, separator).trim();
    const value = part.slice(separator + 1).trim();
    if (!/^\w+$/.test(key) || !value || params.has(key)) {
      throw new ActionParseError(`参数无效或重复：${key}`);
    }
    params.set(key, value);
  }
  return params;
}

function splitTopLevel(text) {
  const parts = [];
  let current = "";
  let quote = null;
  let escaped = false;
  let depth = 0;

  for (const char of text) {
    if (quote) {
      current += char;
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === quote) {
        quote = null;
      }
      continue;
    }
    if (char === "'" || char === '"') {
      quote = char;
      current += char;
    } else if (char === "(" || char === "[") {
      depth += 1;
      current += char;
    } else if (char === ")" || char === "]") {
      depth -= 1;
      if (depth < 0) {
        throw new ActionParseError("参数括号不匹配");
      }
      current += char;
    } else if (char === "," && depth === 0) {
      if (current.trim()) {
        parts.push(current.trim());
      }
      current = "";
    } else {
      current += char;
    }
  }
  if (quote || depth !== 0) {
    throw new ActionParseError("参数中的引号或括号不匹配");
  }
  if (current.trim()) {
    parts.push(current.trim());
  }
  return parts;
}

function findTopLevelEquals(text) {
  let quote = null;
  let escaped = false;
  let depth = 0;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quote) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === quote) {
        quote = null;
      }
    } else if (char === "'" || char === '"') {
      quote = char;
    } else if (char === "(" || char === "[") {
      depth += 1;
    } else if (char === ")" || char === "]") {
      depth -= 1;
    } else if (char === "=" && depth === 0) {
      return index;
    }
  }
  return -1;
}

function parseSingleTextParameter(rawParams, key, optional = false) {
  if (!rawParams && optional) {
    return "";
  }
  const match = rawParams.match(new RegExp(`^${key}\\s*=\\s*([\\s\\S]*)$`));
  if (!match) {
    throw new ActionParseError(`需要唯一的 ${key} 参数`);
  }
  return unquote(match[1].trim());
}

function parsePoint(value) {
  const match = String(value).trim().match(
    /^[\[(]\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*[\])]$/,
  );
  if (!match) {
    throw new ActionParseError(`坐标格式无效：${value}`);
  }
  const point = [Number(match[1]), Number(match[2])];
  if (point.some((coordinate) => !Number.isFinite(coordinate) || coordinate < 0 || coordinate > 999)) {
    throw new ActionParseError(`坐标必须位于 0 到 999：${value}`);
  }
  return point;
}

function parseKeys(value) {
  const text = String(value).trim();
  const inner = text.replace(/^[\[(]\s*/, "").replace(/\s*[\])]$/, "");
  if (!inner) {
    return [];
  }
  return splitTopLevel(inner).map((key) => unquote(key).trim()).filter(Boolean);
}

function parseNumber(value) {
  const number = Number(unquote(String(value)));
  if (!Number.isFinite(number)) {
    throw new ActionParseError(`数字格式无效：${value}`);
  }
  return number;
}

function parseInteger(value) {
  const number = parseNumber(value);
  if (!Number.isInteger(number)) {
    throw new ActionParseError(`需要整数：${value}`);
  }
  return number;
}

function unquote(value) {
  const text = String(value ?? "").trim();
  if (text.length < 2 || !["'", '"'].includes(text[0])) {
    return text;
  }
  const quote = text[0];
  const body = text.endsWith(quote) ? text.slice(1, -1) : text.slice(1);
  return body
    .replaceAll(`\\${quote}`, quote)
    .replaceAll("\\n", "\n")
    .replaceAll("\\r", "\r")
    .replaceAll("\\t", "\t")
    .replaceAll("\\\\", "\\");
}
