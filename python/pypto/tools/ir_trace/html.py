# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Render deterministic, self-contained HTML reports for IR pass traces."""

import json
from pathlib import Path

from .model import PassTrace

_STYLE = """
:root[data-theme="light"] {
  color-scheme: light;
  --bg: #f6f8fa;
  --panel: #ffffff;
  --panel-muted: #eef1f4;
  --text: #1f2328;
  --muted: #59636e;
  --border: #d0d7de;
  --accent: #0969da;
  --accent-text: #ffffff;
  --insert-bg: #dafbe1;
  --delete-bg: #ffebe9;
  --replace-bg: #fff8c5;
  --warning-bg: #fff8c5;
  --warning-border: #d4a72c;
  --keyword: #cf222e;
  --string: #0a3069;
  --number: #0550ae;
  --comment: #6e7781;
  --operator: #8250df;
  --shadow: rgba(31, 35, 40, 0.12);
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0d1117;
  --panel: #161b22;
  --panel-muted: #21262d;
  --text: #e6edf3;
  --muted: #8b949e;
  --border: #30363d;
  --accent: #2f81f7;
  --accent-text: #ffffff;
  --insert-bg: #123820;
  --delete-bg: #491b1f;
  --replace-bg: #3b3213;
  --warning-bg: #3b3213;
  --warning-border: #bb8009;
  --keyword: #ff7b72;
  --string: #a5d6ff;
  --number: #79c0ff;
  --comment: #8b949e;
  --operator: #d2a8ff;
  --shadow: rgba(0, 0, 0, 0.3);
}

* { box-sizing: border-box; }

html, body { min-height: 100%; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

button, input { font: inherit; }

button {
  border: 1px solid var(--border);
  border-radius: 0.35rem;
  background: var(--panel);
  color: var(--text);
  cursor: pointer;
}

button:hover { border-color: var(--accent); }

button:focus-visible, input:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.app {
  display: grid;
  grid-template-columns: 18rem minmax(0, 1fr);
  min-height: 100vh;
}

.sidebar {
  border-right: 1px solid var(--border);
  background: var(--panel);
  min-width: 0;
}

.sidebar-header {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 1rem;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  box-shadow: 0 1px 3px var(--shadow);
}

.sidebar-header h1 {
  margin: 0 0 0.25rem;
  overflow-wrap: anywhere;
  font-size: 1rem;
}

#summary {
  margin: 0 0 0.75rem;
  color: var(--muted);
  font-size: 0.8rem;
}

.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  color: var(--muted);
  font-size: 0.85rem;
}

.filters label { cursor: pointer; }

#pass-list {
  display: grid;
  gap: 0.3rem;
  padding: 0.5rem;
}

.pass-item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.2rem 0.5rem;
  width: 100%;
  padding: 0.55rem 0.65rem;
  text-align: left;
}

.pass-item.selected {
  border-color: var(--accent);
  background: var(--accent);
  color: var(--accent-text);
}

.pass-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pass-stats { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }

.pass-badges {
  grid-column: 1 / -1;
  display: flex;
  gap: 0.35rem;
  font-size: 0.7rem;
}

.badge {
  border: 1px solid currentColor;
  border-radius: 999px;
  padding: 0.05rem 0.35rem;
}

.main {
  min-width: 0;
  padding: 1rem;
}

.toolbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

#pass-title {
  min-width: 12rem;
  margin: 0 auto 0 0;
  font-size: 1.1rem;
}

.toolbar button, .pane-header button { padding: 0.3rem 0.55rem; }

#warnings-panel {
  margin-bottom: 0.75rem;
  padding: 0.7rem;
  border-left: 0.25rem solid var(--warning-border);
  background: var(--warning-bg);
  white-space: pre-wrap;
}

.diff-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 1px;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 0.4rem;
  background: var(--border);
}

.pane {
  min-width: 0;
  background: var(--panel);
}

.pane-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 0.65rem;
  border-bottom: 1px solid var(--border);
  background: var(--panel-muted);
}

.pane-title {
  min-width: 0;
  margin-right: auto;
  overflow: hidden;
  color: var(--muted);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.code-pane {
  overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.78rem;
  line-height: 1.5;
}

.code-line {
  display: grid;
  grid-template-columns: 3.5rem minmax(max-content, 1fr);
  min-height: 1.5em;
}

.code-line.insert { background: var(--insert-bg); }
.code-line.delete { background: var(--delete-bg); }
.code-line.replace { background: var(--replace-bg); }

.line-number {
  padding: 0 0.6rem;
  border-right: 1px solid var(--border);
  color: var(--muted);
  text-align: right;
  user-select: none;
}

.line-code {
  min-width: max-content;
  padding: 0 0.65rem;
  white-space: pre;
}

.fold-button {
  display: block;
  width: calc(100% - 1rem);
  margin: 0.25rem 0.5rem;
  padding: 0.25rem;
  border-style: dashed;
  color: var(--muted);
}

.empty-state { padding: 2rem; color: var(--muted); text-align: center; }
.tok-keyword { color: var(--keyword); font-weight: 600; }
.tok-string { color: var(--string); }
.tok-number { color: var(--number); }
.tok-comment { color: var(--comment); font-style: italic; }
.tok-operator { color: var(--operator); }

@media (max-width: 800px) {
  .app { grid-template-columns: minmax(0, 1fr); }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
  .sidebar-header { position: static; }
  #pass-list { max-height: 14rem; overflow: auto; }
  .main { padding: 0.65rem; }
  .diff-grid { grid-template-columns: minmax(0, 1fr); }
}
"""

_SCRIPT = r"""
"use strict";

const data = JSON.parse(document.getElementById("trace-data").textContent);
document.getElementById("source-name").textContent = data.sourceName;
const changedFilter = document.getElementById("changed-filter");
const noopFilter = document.getElementById("noop-filter");
const passList = document.getElementById("pass-list");
const summary = document.getElementById("summary");
const passTitle = document.getElementById("pass-title");
const beforePane = document.getElementById("before-pane");
const afterPane = document.getElementById("after-pane");
const beforeTitle = document.getElementById("before-title");
const afterTitle = document.getElementById("after-title");
const warningsPanel = document.getElementById("warnings-panel");
const snapshotControls = ["copy-before", "copy-after", "expand-all", "collapse-all"].map((id) =>
  document.getElementById(id)
);
const expandedHunks = new Set();
let selectedIndex = null;

function visiblePasses() {
  return data.passes.filter((trace) =>
    (trace.changed && changedFilter.checked) || (!trace.changed && noopFilter.checked)
  );
}

function closestVisiblePass(passes, index) {
  if (index === null) return passes[0];
  return passes.reduce((closest, trace) => {
    const distance = Math.abs(trace.index - index);
    const closestDistance = Math.abs(closest.index - index);
    // Passes are index-ordered; keeping the first pass makes lower indexes win ties.
    return distance < closestDistance ? trace : closest;
  });
}

function currentTrace() {
  return data.passes.find((trace) => trace.index === selectedIndex);
}

function setSnapshotControlsEnabled(enabled) {
  for (const control of snapshotControls) control.disabled = !enabled;
}

function clearDetail(message) {
  selectedIndex = null;
  passTitle.textContent = message;
  beforeTitle.textContent = "";
  afterTitle.textContent = "";
  warningsPanel.hidden = true;
  warningsPanel.textContent = "";
  beforePane.replaceChildren();
  afterPane.replaceChildren();
  setSnapshotControlsEnabled(false);
}

function selectPass(index) {
  const trace = data.passes.find((candidate) => candidate.index === index);
  if (!trace) return;
  selectedIndex = index;
  setSnapshotControlsEnabled(true);
  renderSidebar();
  renderDiff(trace);
}

function addBadge(container, text) {
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = text;
  container.appendChild(badge);
}

function renderSidebar() {
  const passes = visiblePasses();
  passList.replaceChildren();
  summary.textContent = `${data.changedCount} changed · ${data.noopCount} no-op · ${passes.length} visible`;

  for (const trace of passes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pass-item" + (trace.index === selectedIndex ? " selected" : "");
    button.dataset.index = String(trace.index);
    button.setAttribute("aria-pressed", String(trace.index === selectedIndex));
    button.title = trace.name;
    button.addEventListener("click", () => selectPass(trace.index));

    const name = document.createElement("span");
    name.className = "pass-name";
    name.textContent = `${trace.index}. ${trace.name}`;
    button.appendChild(name);

    const stats = document.createElement("span");
    stats.className = "pass-stats";
    stats.textContent = `+${trace.inserted} -${trace.deleted}`;
    button.appendChild(stats);

    const badges = document.createElement("span");
    badges.className = "pass-badges";
    if (!trace.changed) addBadge(badges, "no-op");
    if (trace.warning) addBadge(badges, "warning");
    button.appendChild(badges);
    passList.appendChild(button);
  }

  if (passes.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No passes match the filters.";
    passList.appendChild(empty);
  }
}

function createCodeLine(side, row) {
  const line = document.createElement("div");
  line.className = `code-line ${row.kind}`;

  const number = document.createElement("span");
  number.className = "line-number";
  number.textContent = row[side + "Number"] ?? "";
  line.appendChild(number);

  const code = document.createElement("span");
  code.className = "line-code";
  code.innerHTML = row[side + "Html"];
  line.appendChild(code);
  return line;
}

function addFoldButton(pane, trace, hunk, hunkIndex) {
  const key = `${trace.index}:${hunkIndex}`;
  const expanded = expandedHunks.has(key);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "fold-button";
  button.textContent = expanded
    ? `Hide ${hunk.rows.length} unchanged lines`
    : `Show ${hunk.rows.length} unchanged lines`;
  button.setAttribute("aria-expanded", String(expanded));
  button.addEventListener("click", () => {
    if (expanded) expandedHunks.delete(key);
    else expandedHunks.add(key);
    renderDiff(trace);
  });
  pane.appendChild(button);
}

function renderDiff(trace) {
  passTitle.textContent = `${trace.index}. ${trace.name} · +${trace.inserted} -${trace.deleted}`;
  beforeTitle.textContent = trace.beforeName;
  afterTitle.textContent = trace.afterName;
  warningsPanel.hidden = !trace.warning;
  warningsPanel.textContent = trace.warning || "";
  beforePane.replaceChildren();
  afterPane.replaceChildren();

  trace.hunks.forEach((hunk, hunkIndex) => {
    const key = `${trace.index}:${hunkIndex}`;
    const expanded = expandedHunks.has(key);
    if (hunk.collapsed) {
      addFoldButton(beforePane, trace, hunk, hunkIndex);
      addFoldButton(afterPane, trace, hunk, hunkIndex);
      if (!expanded) return;
    }
    for (const row of hunk.rows) {
      beforePane.appendChild(createCodeLine("before", row));
      afterPane.appendChild(createCodeLine("after", row));
    }
  });
}

function fallbackCopy(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function copySnapshot(side) {
  const trace = currentTrace();
  if (!trace) return;
  const text = trace[side + "Text"];
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
  } else {
    fallbackCopy(text);
  }
}

function setAllHunks(expanded) {
  const trace = currentTrace();
  if (!trace) return;
  trace.hunks.forEach((hunk, hunkIndex) => {
    if (!hunk.collapsed) return;
    const key = `${trace.index}:${hunkIndex}`;
    if (expanded) expandedHunks.add(key);
    else expandedHunks.delete(key);
  });
  renderDiff(trace);
}

function toggleTheme() {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
}

function applyFilters() {
  const passes = visiblePasses();
  if (!passes.some((trace) => trace.index === selectedIndex)) {
    if (passes[0]) selectPass(closestVisiblePass(passes, selectedIndex).index);
    else {
      clearDetail("No passes match the filters.");
      renderSidebar();
    }
  } else {
    renderSidebar();
  }
}

changedFilter.addEventListener("change", applyFilters);
noopFilter.addEventListener("change", applyFilters);
document.getElementById("copy-before").addEventListener("click", () => copySnapshot("before"));
document.getElementById("copy-after").addEventListener("click", () => copySnapshot("after"));
document.getElementById("expand-all").addEventListener("click", () => setAllHunks(true));
document.getElementById("collapse-all").addEventListener("click", () => setAllHunks(false));
document.getElementById("theme-toggle").addEventListener("click", toggleTheme);

document.addEventListener("keydown", (event) => {
  const target = event.target;
  if (target && (target.tagName === "INPUT" || target.tagName === "BUTTON")) return;
  const passes = visiblePasses();
  if (passes.length === 0) return;
  const current = passes.findIndex((trace) => trace.index === selectedIndex);
  let next = current < 0 ? 0 : current;
  if (event.key === "j" || event.key === "ArrowDown") next = Math.min(next + 1, passes.length - 1);
  else if (event.key === "k" || event.key === "ArrowUp") next = Math.max(next - 1, 0);
  else return;
  event.preventDefault();
  selectPass(passes[next].index);
});

document.documentElement.dataset.theme = window.matchMedia("(prefers-color-scheme: dark)").matches
  ? "dark"
  : "light";
const initialTrace = data.passes.find((trace) => trace.changed) || data.passes[0];
if (initialTrace) selectPass(initialTrace.index);
else {
  clearDetail("No passes in this report.");
  renderSidebar();
}
"""

_BODY = """
<div class="app">
  <aside class="sidebar">
    <div class="sidebar-header">
      <h1 id="source-name"></h1>
      <p id="summary"></p>
      <div class="filters" aria-label="Pass filters">
        <label><input id="changed-filter" type="checkbox" checked> Changed</label>
        <label><input id="noop-filter" type="checkbox" checked> No-op</label>
      </div>
    </div>
    <nav id="pass-list" aria-label="IR passes"></nav>
  </aside>
  <main class="main">
    <div class="toolbar">
      <h2 id="pass-title">No passes</h2>
      <button id="expand-all" type="button">Expand all</button>
      <button id="collapse-all" type="button">Collapse all</button>
      <button id="theme-toggle" type="button">Toggle theme</button>
    </div>
    <section id="warnings-panel" aria-label="Pass warnings" hidden></section>
    <div class="diff-grid">
      <section class="pane" aria-label="Before snapshot">
        <header class="pane-header">
          <span id="before-title" class="pane-title"></span>
          <button id="copy-before" type="button">Copy full source</button>
        </header>
        <div id="before-pane" class="code-pane"></div>
      </section>
      <section class="pane" aria-label="After snapshot">
        <header class="pane-header">
          <span id="after-title" class="pane-title"></span>
          <button id="copy-after" type="button">Copy full source</button>
        </header>
        <div id="after-pane" class="code-pane"></div>
      </section>
    </div>
  </main>
</div>
"""


def _trace_payload(traces: tuple[PassTrace, ...], source_name: str) -> dict[str, object]:
    """Build the stable, path-free payload consumed by the report."""
    passes = []
    for trace in traces:
        passes.append(
            {
                "afterName": trace.after.path.name,
                "afterText": trace.after.text,
                "beforeName": trace.before.path.name,
                "beforeText": trace.before.text,
                "changed": trace.changed,
                "deleted": trace.deleted,
                "hunks": [
                    {
                        "collapsed": hunk.collapsed,
                        "rows": [
                            {
                                "afterHtml": row.after_html,
                                "afterNumber": row.after_number,
                                "beforeHtml": row.before_html,
                                "beforeNumber": row.before_number,
                                "kind": row.kind,
                            }
                            for row in hunk.rows
                        ],
                    }
                    for hunk in trace.hunks
                ],
                "index": trace.index,
                "inserted": trace.inserted,
                "name": trace.name,
                "warning": trace.after.warning_text,
            }
        )
    changed_count = sum(trace.changed for trace in traces)
    return {
        "changedCount": changed_count,
        "noopCount": len(traces) - changed_count,
        "passes": passes,
        "sourceName": Path(source_name).name,
    }


def _json_for_script(payload: object) -> str:
    """Encode JSON without sequences that can terminate an HTML script element."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    escapes = {ord("<"): "\\u003c", ord(">"): "\\u003e", ord("&"): "\\u0026"}
    escapes.update({0x2028: "\\u2028", 0x2029: "\\u2029"})
    return encoded.translate(escapes)


def render_html(traces: tuple[PassTrace, ...], source_name: str) -> str:
    """Return a deterministic, self-contained HTML5 IR trace report."""
    payload = _json_for_script(_trace_payload(traces, source_name))
    return (
        "<!doctype html>\n"
        '<html lang="en" data-theme="light">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>IR pass trace</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        f"<body>{_BODY}\n"
        f'<script id="trace-data" type="application/json">{payload}</script>\n'
        f"<script>{_SCRIPT}</script>\n"
        "</body>\n"
        "</html>\n"
    )
