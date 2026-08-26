export function normalizeHotkeyKeys(keys, platformOs) {
  const isMac = String(platformOs || "").toLowerCase() === "mac";
  return (Array.isArray(keys) ? keys : []).map((rawKey) => {
    const key = String(rawKey).toLowerCase();
    if (key === "cmd" || key === "command") return "meta";
    if (key === "controlormeta") return isMac ? "meta" : "ctrl";
    // Models commonly emit `ctrl` for platform-neutral editing shortcuts.
    // Preserve `control` as the spelling for the physical Control key on Mac.
    if (key === "ctrl") return isMac ? "meta" : "ctrl";
    if (key === "control") return "ctrl";
    if (key === "option") return "alt";
    return key;
  });
}

export function editingCommandsForHotkey(keys, platformOs) {
  const normalized = Array.isArray(keys) ? keys : [];
  const primaryModifier = String(platformOs || "").toLowerCase() === "mac" ? "meta" : "ctrl";
  const modifiers = normalized.filter((key) => ["alt", "ctrl", "meta", "shift"].includes(key));
  const primaryKeys = normalized.filter((key) => !["alt", "ctrl", "meta", "shift"].includes(key));

  if (
    primaryKeys.length === 1
    && primaryKeys[0] === "a"
    && modifiers.length === 1
    && modifiers[0] === primaryModifier
  ) {
    return ["selectAll"];
  }
  return [];
}
