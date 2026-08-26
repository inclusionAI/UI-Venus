const MIME_BY_EXTENSION = new Map([
  [".csv", "text/csv"],
  [".gif", "image/gif"],
  [".jpeg", "image/jpeg"],
  [".jpg", "image/jpeg"],
  [".json", "application/json"],
  [".pdf", "application/pdf"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".txt", "text/plain"],
  [".webp", "image/webp"],
  [".xml", "application/xml"],
  [".zip", "application/zip"],
]);

export function findWorkspaceFile(files, requestedFile, accept = "") {
  const requested = normalizeWorkspaceRelativePath(requestedFile);
  const candidates = listAcceptedWorkspaceFiles(files, accept);
  const exact = candidates.find((file) => normalizeSlashes(file.relativePath) === requested);
  if (exact) return exact;

  const basenameMatches = candidates.filter(
    (file) => normalizeSlashes(file.relativePath).split("/").at(-1) === requested,
  );
  return basenameMatches.length === 1 ? basenameMatches[0] : null;
}

export function listAcceptedWorkspaceFiles(files, accept = "") {
  return (Array.isArray(files) ? files : [])
    .filter(isUsableFile)
    .filter((file) => acceptsFile(file, accept));
}

export function normalizeWorkspaceRelativePath(value) {
  const path = normalizeSlashes(String(value ?? "").trim()).replace(/^\.\//, "");
  if (!path || path.startsWith("/") || path.split("/").some((part) => !part || part === "." || part === "..")) {
    throw new Error("workspace 文件必须是安全的相对路径");
  }
  return path;
}

export function normalizeDownloadFilename(value) {
  const filename = String(value ?? "").trim();
  if (!filename || filename === "." || filename === ".." || /[\\/\0]/.test(filename)) {
    throw new Error("下载文件名不能包含路径或空字符");
  }
  return filename;
}

export function normalizeDownloadResourceUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    if (url.hostname !== "github.com") return url.href;
    const parts = url.pathname.split("/").filter(Boolean);
    const markerIndex = parts.findIndex((part) => part === "blob" || part === "raw");
    if (parts.length >= 5 && markerIndex === 2) {
      const [owner, repository, , revision, ...path] = parts;
      if (revision && path.length) {
        const raw = new URL(`https://raw.githubusercontent.com/${owner}/${repository}/${revision}/${path.join("/")}`);
        raw.search = url.search;
        return raw.href;
      }
    }
    return url.href;
  } catch {
    return String(rawUrl || "");
  }
}

export function githubFileDownloadTarget(rawUrl) {
  try {
    const url = new URL(rawUrl);
    if (url.hostname !== "github.com") return null;
    const parts = url.pathname.split("/").filter(Boolean);
    const markerIndex = parts.findIndex((part) => part === "blob" || part === "raw");
    if (markerIndex !== 2 || parts.length < 5) return null;
    const encodedName = parts.at(-1) || "";
    let suggestedFilename = encodedName;
    try {
      suggestedFilename = decodeURIComponent(encodedName);
    } catch {
      // Keep the URL-encoded basename if it is malformed.
    }
    return { url: url.href, suggestedFilename };
  } catch {
    return null;
  }
}

export function stabilizeDownloadUrl(downloadUrl, pageUrl) {
  try {
    if (new URL(downloadUrl).protocol !== "blob:") return downloadUrl;
  } catch {
    return downloadUrl;
  }
  return githubFileDownloadTarget(pageUrl)?.url || downloadUrl;
}

export function isLikelyDownloadLink(target) {
  if (!target?.url) return false;
  if (target.hasDownloadAttribute) return true;
  const mimeType = String(target.mimeType ?? "").toLowerCase();

  let parsedUrl;
  try {
    parsedUrl = new URL(target.url);
  } catch {
    return false;
  }
  const pathname = parsedUrl.pathname.toLowerCase();
  // A GitHub /blob/ URL opens the repository's file viewer. Its extension
  // describes the viewed file, not an immediate download action.
  if (parsedUrl.hostname === "github.com" && /\/blob\//i.test(pathname)) return false;
  if (mimeType === "application/pdf" || mimeType === "application/octet-stream") return true;
  if (/\/(?:pdf|download|e-print)(?:\/|$)/.test(pathname)) return true;
  if (/\.(?:csv|docx?|epub|gif|jpe?g|json|odp|ods|odt|pdf|png|pptx?|tar|tgz|txt|webp|xlsx?|xml|zip)$/i.test(pathname)) {
    return true;
  }
  return /\b(?:download|pdf|full[ -]?text)\b/i.test(String(target.text ?? ""));
}

function isUsableFile(file) {
  return file
    && typeof file.relativePath === "string"
    && file.relativePath.length > 0;
}

function acceptsFile(file, rawAccept) {
  const accepts = String(rawAccept ?? "")
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  if (accepts.length === 0) return true;

  const relativePath = file.relativePath.toLowerCase();
  const extension = extensionOf(relativePath);
  const mimeType = String(file.mimeType || MIME_BY_EXTENSION.get(extension) || "application/octet-stream")
    .toLowerCase();
  return accepts.some((accept) => {
    if (accept.startsWith(".")) return relativePath.endsWith(accept);
    if (accept.endsWith("/*")) return mimeType.startsWith(accept.slice(0, -1));
    return mimeType === accept;
  });
}

function extensionOf(path) {
  const basename = path.split("/").at(-1) || "";
  const dot = basename.lastIndexOf(".");
  return dot >= 0 ? basename.slice(dot).toLowerCase() : "";
}

function normalizeSlashes(value) {
  return String(value).replaceAll("\\", "/");
}
