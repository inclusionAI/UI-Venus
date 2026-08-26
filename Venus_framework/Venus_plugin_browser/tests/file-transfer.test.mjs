import test from "node:test";
import assert from "node:assert/strict";

import {
  findWorkspaceFile,
  githubFileDownloadTarget,
  isLikelyDownloadLink,
  listAcceptedWorkspaceFiles,
  normalizeDownloadFilename,
  normalizeDownloadResourceUrl,
  normalizeWorkspaceRelativePath,
  stabilizeDownloadUrl,
} from "../src/file-transfer.js";

const files = [
  {
    relativePath: "photo.png",
    mimeType: "image/png",
  },
  {
    relativePath: "report.pdf",
    mimeType: "application/pdf",
  },
];

test("uses the exact model-selected upload and honors accept", () => {
  assert.equal(findWorkspaceFile(files, "report.pdf")?.relativePath, "report.pdf");
  assert.equal(findWorkspaceFile(files, "report.pdf", "image/*"), null);
  assert.equal(findWorkspaceFile(files, "photo.png", "image/*")?.relativePath, "photo.png");
  assert.deepEqual(
    listAcceptedWorkspaceFiles(files, ".pdf,application/pdf").map((file) => file.relativePath),
    ["report.pdf"],
  );
});

test("rejects unsafe upload paths and download filenames", () => {
  assert.equal(normalizeWorkspaceRelativePath("./folder/report.pdf"), "folder/report.pdf");
  assert.equal(normalizeDownloadFilename("final-report.pdf"), "final-report.pdf");
  assert.throws(() => normalizeWorkspaceRelativePath("../secret.txt"), /安全/);
  assert.throws(() => normalizeDownloadFilename("folder/report.pdf"), /路径/);
});

test("recognizes inline arXiv PDF links as downloads", () => {
  assert.equal(isLikelyDownloadLink({
    url: "https://arxiv.org/pdf/2404.14068",
    text: "View PDF",
  }), true);
  assert.equal(isLikelyDownloadLink({
    url: "https://example.com/articles/123",
    text: "Read article",
  }), false);
});

test("treats GitHub blob file links as navigation rather than downloads", () => {
  assert.equal(isLikelyDownloadLink({
    url: "https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5/UI_Venus_1_5_Technical_Report.pdf",
    text: "UI_Venus_1_5_Technical_Report.pdf",
  }), false);
  assert.equal(isLikelyDownloadLink({
    url: "https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5/report.pdf",
    text: "Download report",
    hasDownloadAttribute: true,
  }), true);
});

test("converts GitHub blob and raw links to the raw content host", () => {
  assert.equal(
    normalizeDownloadResourceUrl("https://github.com/openai/example/blob/main/README.md"),
    "https://raw.githubusercontent.com/openai/example/main/README.md",
  );
  assert.equal(
    normalizeDownloadResourceUrl("https://github.com/openai/example/raw/main/docs/file.txt?download=1"),
    "https://raw.githubusercontent.com/openai/example/main/docs/file.txt?download=1",
  );
  assert.equal(
    normalizeDownloadResourceUrl("https://example.com/file.txt"),
    "https://example.com/file.txt",
  );
});

test("infers a downloadable file from a GitHub file page", () => {
  assert.deepEqual(
    githubFileDownloadTarget("https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5/README.md"),
    {
      url: "https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5/README.md",
      suggestedFilename: "README.md",
    },
  );
  assert.equal(githubFileDownloadTarget("https://github.com/inclusionAI/UI-Venus"), null);
  assert.equal(githubFileDownloadTarget("https://example.com/blob/main/file.txt"), null);
});

test("replaces a GitHub page blob download with its stable file URL", () => {
  assert.equal(
    stabilizeDownloadUrl(
      "blob:https://github.com/532d1dd5-c92e-4b18-8f0d-dfb5a83d576b",
      "https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5/assets/ui-venus-logo-3.png",
    ),
    "https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5/assets/ui-venus-logo-3.png",
  );
  assert.equal(
    stabilizeDownloadUrl("https://example.com/image.png", "https://github.com/example/repo/blob/main/image.png"),
    "https://example.com/image.png",
  );
});
