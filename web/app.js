const API_BASE_URL = "http://127.0.0.1:8000";
const SESSION_KEY = "rag_assistant_session_id";

const messagesElement = document.querySelector("#messages");
const welcomeCard = document.querySelector("#welcome-card");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const newChatButton = document.querySelector("#new-chat-button");
const chatList = document.querySelector("#chat-list");
const statusElement = document.querySelector("#connection-status");
let activeSessionId = localStorage.getItem(SESSION_KEY);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addMessage(role, content, sources = []) {
  welcomeCard?.remove();

  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  const label = role === "user" ? "You" : "R";
  const sourceMarkup = sources.length
    ? `<details class="sources"><summary>${sources.length} source${sources.length === 1 ? "" : "s"}</summary>${sources.map((source) => `
        <div class="source-item"><strong>${escapeHtml(source.source || "Document")}</strong>${source.page ? ` · page ${escapeHtml(source.page)}` : ""}<br>${escapeHtml(source.data || "")}</div>`).join("")}</details>`
    : "";

  row.innerHTML = `
    <div class="avatar" aria-hidden="true">${label}</div>
    <div class="bubble">${escapeHtml(content)}${sourceMarkup}</div>
  `;
  messagesElement.appendChild(row);
  messagesElement.scrollTop = messagesElement.scrollHeight;
}

function clearMessages() {
  messagesElement.innerHTML = `
    <div class="welcome-card" id="welcome-card">
      <div class="welcome-icon">✦</div>
      <h3>Welcome to your documentation assistant</h3>
      <p>Ask a question and I’ll search your indexed documents for a useful answer.</p>
    </div>
  `;
}

function addTyping() {
  const row = document.createElement("div");
  row.className = "message-row assistant typing";
  row.innerHTML = '<div class="avatar">R</div><div class="bubble">Searching your documents…</div>';
  messagesElement.appendChild(row);
  messagesElement.scrollTop = messagesElement.scrollHeight;
  return row;
}

function setBusy(busy) {
  input.disabled = busy;
  sendButton.disabled = busy;
  statusElement.textContent = busy ? "Thinking…" : "Ready";
}

async function loadHistory() {
  const sessionId = activeSessionId;
  if (!sessionId) return;

  try {
    const response = await fetch(`${API_BASE_URL}/chat/${encodeURIComponent(sessionId)}`);
    if (!response.ok) {
      localStorage.removeItem(SESSION_KEY);
      return;
    }
    const data = await response.json();
    if (!data.messages?.length) return;
    welcomeCard?.remove();
    data.messages.forEach((message) => addMessage(message.role, message.content));
  } catch (error) {
    statusElement.textContent = "API unavailable";
  }
}

async function loadSession(sessionId) {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/${encodeURIComponent(sessionId)}`);
    if (!response.ok) throw new Error("Unable to load chat");
    const data = await response.json();
    activeSessionId = sessionId;
    localStorage.setItem(SESSION_KEY, sessionId);
    clearMessages();
    data.messages.forEach((message) => addMessage(message.role, message.content));
    renderChatList();
  } catch (error) {
    statusElement.textContent = error.message;
  }
}

async function renderChatList() {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/sessions`);
    if (!response.ok) throw new Error("Unable to load chats");
    const data = await response.json();
    chatList.innerHTML = "";
    if (!data.sessions.length) {
      chatList.innerHTML = '<div class="chat-list-empty">No previous chats</div>';
      return;
    }
    data.sessions.forEach((session) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = session.session_id === activeSessionId ? "active" : "";
      button.textContent = session.preview;
      button.title = session.preview;
      button.addEventListener("click", () => loadSession(session.session_id));
      chatList.appendChild(button);
    });
  } catch (error) {
    chatList.innerHTML = '<div class="chat-list-empty">Unable to load chats</div>';
  }
}

async function createNewChat() {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/session`, { method: "POST" });
    if (!response.ok) throw new Error("Unable to create chat");
    const data = await response.json();
    activeSessionId = data.session_id;
    localStorage.setItem(SESSION_KEY, activeSessionId);
    clearMessages();
    renderChatList();
    input.focus();
  } catch (error) {
    statusElement.textContent = error.message;
  }
}

async function sendMessage(message) {
  const sessionId = activeSessionId;
  addMessage("user", message);
  const typing = addTyping();
  setBusy(true);

  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, ...(sessionId ? { session_id: sessionId } : {}) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Chat request failed");

    activeSessionId = data.session_id;
    localStorage.setItem(SESSION_KEY, activeSessionId);
    typing.remove();
    addMessage("assistant", data.answer, data.sources || []);
    renderChatList();
  } catch (error) {
    typing.remove();
    addMessage("assistant", `Sorry, I couldn't complete that request. ${error.message}`);
    statusElement.classList.add("error");
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || sendButton.disabled) return;
  input.value = "";
  input.style.height = "auto";
  sendMessage(message);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
});

newChatButton.addEventListener("click", () => {
  createNewChat();
});

document.querySelectorAll("[data-suggestion]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.suggestion;
    input.focus();
  });
});

renderChatList();
loadHistory();
