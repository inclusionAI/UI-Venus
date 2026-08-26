export function missingConfigurationFields(settings) {
  const missing = [];
  if (!String(settings?.apiUrl || "").trim()) missing.push("API URL");
  if (!String(settings?.apiKey || "").trim()) missing.push("API Key");
  if (!String(settings?.model || "").trim()) missing.push("Model");
  return missing;
}

export function missingConfigurationMessage(fields) {
  const items = Array.isArray(fields) ? fields.filter(Boolean) : [];
  return items.length ? `请先完成以下设置：${items.join("、")}` : "";
}
