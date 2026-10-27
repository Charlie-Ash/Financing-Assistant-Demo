// Chat page logic: talks to the JSON API in web/server.py. Plain JavaScript,
// no build step or external libraries.
"use strict";

const AGENT_LABELS = { rag: "法規查詢", credit_lookup: "客戶查詢", default: "一般對話" };
const ROLE_LABELS = { sales: "業務", credit: "授信" };
const AGENT_HELP = [
  { name: "rag", title: "法規查詢", text: "回答動產擔保交易法、消費者保護法、個人資料保護法、洗錢防制法與公司內部規範的問題，並附上條號出處。",
    examples: ["提前清償要付違約金嗎？", "個資法規定蒐集個人資料時要告知哪些事項？"] },
  { name: "credit_lookup", title: "客戶查詢", text: "查詢客戶的合約、繳款與逾期紀錄，依你的身分顯示可看的資料。",
    examples: ["王大明有沒有逾期紀錄？", "查一下 C0003 的合約", "他最近一年繳款正常嗎？"] },
  { name: "default", title: "一般對話", text: "一般問題與聊天。", examples: ["你好，你可以幫我做什麼？"] },
];

const $ = (selector) => document.querySelector(selector);
let connectedAgents = [];
let pending = false;

// ---------- API ----------

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? "輸入格式不正確。" : (data.detail || `伺服器錯誤（${response.status}）`);
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return data;
}

// ---------- Markdown (escape first, then format -- reply text can never inject HTML) ----------

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function formatInline(escaped) {
  return escaped
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(https?:\/\/[^\s<）)」]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split(/\r?\n/);
  const html = [];
  let list = null;       // "ul" | "ol" | null
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) html.push(`<p>${paragraph.map(formatInline).join("<br>")}</p>`);
    paragraph = [];
  };
  const closeList = () => { if (list) html.push(`</${list}>`); list = null; };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*(?:[-*•])\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const heading = line.match(/^(#{1,4})\s+(.*)$/);

    if (bullet || numbered) {
      flushParagraph();
      const type = bullet ? "ul" : "ol";
      if (list !== type) { closeList(); html.push(`<${type}>`); list = type; }
      html.push(`<li>${formatInline((bullet || numbered)[1])}</li>`);
    } else if (heading) {
      flushParagraph(); closeList();
      html.push(`<h4>${formatInline(heading[2])}</h4>`);
    } else if (line.trim() === "") {
      flushParagraph(); closeList();
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  flushParagraph(); closeList();
  return html.join("");
}

// ---------- Messages ----------

function addMessage(entry) {
  $("#intro").hidden = true;
  const node = $("#message-template").content.firstElementChild.cloneNode(true);
  node.classList.add(entry.role);
  const meta = node.querySelector(".message-meta");
  const body = node.querySelector(".message-body");
  const actions = node.querySelector(".message-actions");

  if (entry.role === "user") {
    body.textContent = entry.text;
    meta.remove(); actions.remove();
  } else if (entry.role === "error") {
    meta.textContent = "錯誤";
    body.textContent = entry.text;
    actions.remove();
  } else {
    const tag = document.createElement("span");
    tag.className = "agent-tag";
    tag.textContent = AGENT_LABELS[entry.agent] || entry.agent;
    meta.append(tag);
    if (entry.elapsed_s !== undefined) {
      const time = document.createElement("span");
      time.textContent = `${entry.elapsed_s} 秒`;
      meta.append(time);
    }
    body.innerHTML = renderMarkdown(entry.text);
    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "複製";
    copy.addEventListener("click", () => copyText(entry.text, copy));
    actions.append(copy);
  }

  $("#messages").append(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

function addThinking() {
  const node = $("#message-template").content.firstElementChild.cloneNode(true);
  node.classList.add("assistant");
  node.querySelector(".message-meta").remove();
  node.querySelector(".message-actions").remove();
  node.querySelector(".message-body").innerHTML = '<span class="thinking"><span class="spinner"></span>思考中…</span>';
  $("#messages").append(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Clipboard API needs https or localhost; fall back to a hidden textarea.
    const area = document.createElement("textarea");
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  button.textContent = "已複製";
  setTimeout(() => { button.textContent = "複製"; }, 1500);
}

function clearMessages() {
  for (const node of [...$("#messages").querySelectorAll(".message")]) node.remove();
  $("#intro").hidden = false;
}

// ---------- Screens ----------

function showWelcome() {
  clearMessages();
  $("#chat-screen").hidden = true;
  $("#welcome-screen").hidden = false;
  $("#start-error").hidden = true;
  $("#start-button").disabled = false;
  $("#user-id").focus();
}

function showChat(session) {
  connectedAgents = session.connected_agents || [];
  $("#role-badge").textContent = `${session.role_label}｜${session.caller.user_id}`;
  clearMessages();
  for (const entry of session.transcript || []) addMessage(entry);
  $("#welcome-screen").hidden = true;
  $("#chat-screen").hidden = false;
  $("#chat-input").focus();
}

async function startChat(event) {
  event.preventDefault();
  const userId = $("#user-id").value.trim();
  const role = new FormData($("#start-form")).get("role");
  if (!userId || /\s/.test(userId)) {
    showStartError("請輸入使用者代號（不可包含空白）。");
    return;
  }
  $("#start-button").disabled = true;
  $("#start-button").textContent = "連線中…";
  try {
    showChat(await api("/api/session", { method: "POST", body: { user_id: userId, role } }));
  } catch (error) {
    if (error.status === 409) {
      showChat(await api("/api/session"));  // a chat is already open -- resume it
    } else {
      showStartError(error.message);
    }
  } finally {
    $("#start-button").disabled = false;
    $("#start-button").textContent = "開始對話";
  }
}

function showStartError(message) {
  $("#start-error").textContent = message;
  $("#start-error").hidden = false;
}

// ---------- Sending ----------

async function sendMessage(text) {
  text = text.trim();
  if (!text || pending) return;
  pending = true;
  $("#send-button").disabled = true;
  $("#chat-input").value = "";
  autoResize();

  addMessage({ role: "user", text });
  const thinking = addThinking();
  try {
    const entry = await api("/api/chat", { method: "POST", body: { message: text } });
    thinking.remove();
    addMessage(entry);
  } catch (error) {
    thinking.remove();
    if (error.status === 409) { showWelcome(); return; }  // the chat ended elsewhere
    addMessage({ role: "error", text: error.message });
  } finally {
    pending = false;
    $("#send-button").disabled = false;
    $("#chat-input").focus();
  }
}

function autoResize() {
  const input = $("#chat-input");
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

// ---------- Help ----------

function openHelp() {
  const list = $("#agent-list");
  list.replaceChildren();
  const chatActive = !$("#chat-screen").hidden;
  for (const agent of AGENT_HELP) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = agent.title;
    item.append(title);
    if (agent.name !== "default" && chatActive) {
      const ok = connectedAgents.includes(agent.name);
      const status = document.createElement("span");
      status.className = `status ${ok ? "ok" : "down"}`;
      status.textContent = ok ? "● 已連線" : "● 未連線";
      item.append(status);
    }
    const text = document.createElement("div");
    text.textContent = agent.text;
    const examples = document.createElement("div");
    examples.className = "examples";
    examples.textContent = `例如：${agent.examples.join("、")}`;
    item.append(text, examples);
    list.append(item);
  }
  $("#help-dialog").showModal();
}

// ---------- Ending a chat ----------

function showEndStep(step) {
  for (const node of $("#end-dialog").querySelectorAll("[data-step]")) node.hidden = node.dataset.step !== step;
}

function openEndDialog() {
  if (pending) return;
  showEndStep("confirm");
  $("#end-dialog").showModal();
}

async function endChat(save) {
  showEndStep("working");
  try {
    const result = await api("/api/session/end", { method: "POST", body: { save } });
    if (result.saved) {
      $("#saved-name").textContent = result.file_name;
      $("#download-link").href = result.download_url;
      $("#download-link").setAttribute("download", result.file_name);
      showEndStep("done");
    } else {
      $("#end-dialog").close();
      showWelcome();
    }
  } catch (error) {
    $("#end-error").textContent = error.message;
    showEndStep("error");
  }
}

// ---------- Wiring ----------

function wire() {
  $("#start-form").addEventListener("submit", startChat);
  $("#chat-form").addEventListener("submit", (event) => { event.preventDefault(); sendMessage($("#chat-input").value); });
  $("#chat-input").addEventListener("input", autoResize);
  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage($("#chat-input").value);
    }
  });
  for (const chip of document.querySelectorAll("[data-example]")) {
    chip.addEventListener("click", () => sendMessage(chip.textContent));
  }
  for (const button of document.querySelectorAll("[data-open-help]")) button.addEventListener("click", openHelp);
  for (const button of document.querySelectorAll("[data-close]")) {
    button.addEventListener("click", () => button.closest("dialog").close());
  }
  $("#end-button").addEventListener("click", openEndDialog);
  $("#end-save").addEventListener("click", () => endChat(true));
  $("#end-discard").addEventListener("click", () => endChat(false));
  $("#new-chat").addEventListener("click", () => { $("#end-dialog").close(); showWelcome(); });
  // Closing the "saved" dialog any other way (Esc) also returns to the welcome screen.
  $("#end-dialog").addEventListener("close", () => {
    if (!$("#end-dialog").querySelector('[data-step="done"]').hidden) showWelcome();
  });
}

async function init() {
  wire();
  try {
    const session = await api("/api/session");
    if (session.active) showChat(session); else showWelcome();
  } catch {
    showWelcome();
    showStartError("無法連線到伺服器，請確認 web.server 是否已啟動。");
  }
}

init();
