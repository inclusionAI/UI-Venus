import test from "node:test";
import assert from "node:assert/strict";

import {
  missingConfigurationFields,
  missingConfigurationMessage,
} from "../src/config-validation.js";

test("lists missing model settings without requiring a workspace", () => {
  const missing = missingConfigurationFields({});
  assert.deepEqual(missing, ["API URL", "API Key", "Model"]);
  assert.equal(
    missingConfigurationMessage(missing),
    "请先完成以下设置：API URL、API Key、Model",
  );
});

test("does not require workspace configuration for ordinary tasks", () => {
  const settings = { apiUrl: "https://api.example.com", apiKey: "key", model: "vision" };
  assert.deepEqual(missingConfigurationFields(settings), []);
});
