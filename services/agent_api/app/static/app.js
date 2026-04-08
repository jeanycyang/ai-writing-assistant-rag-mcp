const state = {
  sessions: [],
  activeSessionId: null,
  activeSession: null,
  responsePayloads: new Map(),
};

const sessionListEl = document.getElementById("session-list");
const sessionTitleEl = document.getElementById("session-title");
const sessionMetaEl = document.getElementById("session-meta");
const messagesEl = document.getElementById("messages");
const statusEl = document.getElementById("status");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("message-input");
const timingEl = document.getElementById("include-timing");
const newSessionButtonEl = document.getElementById("new-session-button");
const RESPONSE_STORAGE_KEY = "fanfiction-rag-response-payloads";

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function formatTimestamp(value) {
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function responseStorageId(sessionId, createdAt) {
  return `${sessionId}:${createdAt}`;
}

function saveResponsePayloads() {
  const serializable = Object.fromEntries(state.responsePayloads.entries());
  localStorage.setItem(RESPONSE_STORAGE_KEY, JSON.stringify(serializable));
}

function loadResponsePayloads() {
  try {
    const raw = localStorage.getItem(RESPONSE_STORAGE_KEY);
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      state.responsePayloads = new Map(Object.entries(parsed));
    }
  } catch {
    state.responsePayloads = new Map();
  }
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const helper = document.createElement("textarea");
  helper.value = value;
  helper.setAttribute("readonly", "");
  helper.style.position = "absolute";
  helper.style.left = "-9999px";
  document.body.appendChild(helper);
  helper.select();
  document.execCommand("copy");
  document.body.removeChild(helper);
}

function renderSessions() {
  sessionListEl.innerHTML = "";
  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item${session.session_id === state.activeSessionId ? " active" : ""}`;
    button.innerHTML = `
      <div class="session-title">${escapeHtml(session.title || "New Session")}</div>
      <div class="session-updated">${escapeHtml(formatTimestamp(session.updated_at))}</div>
    `;
    button.addEventListener("click", () => loadSession(session.session_id));
    sessionListEl.appendChild(button);
  }
}

function renderMessages() {
  const active = state.activeSession;
  messagesEl.innerHTML = "";
  if (!active || active.messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Start a session and ask a canon question.";
    messagesEl.appendChild(empty);
    return;
  }

  for (const message of active.messages) {
    const wrapper = document.createElement("article");
    wrapper.className = `message ${message.role}`;
    wrapper.innerHTML = `
      <div>${escapeHtml(message.content).replaceAll("\n", "<br />")}</div>
      <div class="message-meta">${escapeHtml(formatTimestamp(message.created_at))}</div>
    `;

    if (message.role === "assistant") {
      const payloadKey = responseStorageId(state.activeSessionId, message.created_at);
      const payload = state.responsePayloads.get(payloadKey);
      if (payload) {
        const actions = document.createElement("div");
        actions.className = "assistant-actions";
        const copyButton = document.createElement("button");
        copyButton.type = "button";
        copyButton.className = "copy-response-button";
        copyButton.textContent = "Copy HTTP Response";
        copyButton.addEventListener("click", async () => {
          try {
            await copyText(JSON.stringify(payload, null, 2));
            setStatus("Copied HTTP response.");
          } catch (error) {
            setStatus(error.message || "Failed to copy HTTP response.", true);
          }
        });
        actions.appendChild(copyButton);
        wrapper.appendChild(actions);

        const extras = document.createElement("div");
        extras.className = "assistant-extras";
        extras.innerHTML = `
          <details>
            <summary>Model Inputs</summary>
            <div class="prompt-block">${escapeHtml(JSON.stringify(payload.debug?.model_inputs || [], null, 2))}</div>
          </details>
          <details>
            <summary>Citations</summary>
            <div class="citations">${escapeHtml(JSON.stringify(payload.citations, null, 2))}</div>
          </details>
          <details>
            <summary>Debug</summary>
            <div class="debug-block">${escapeHtml(JSON.stringify(payload.debug, null, 2))}</div>
          </details>
        `;
        wrapper.appendChild(extras);
      }
    }

    messagesEl.appendChild(wrapper);
  }

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderActiveSession() {
  const active = state.activeSession;
  sessionTitleEl.textContent = active?.title || "New Session";
  sessionMetaEl.textContent = active
    ? `Updated ${formatTimestamp(active.updated_at)}`
    : "Local-only in-memory chat";
  renderMessages();
  renderSessions();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // Ignore non-JSON errors.
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

async function refreshSessions() {
  const payload = await fetchJson("/sessions");
  state.sessions = payload.sessions;
  renderSessions();
}

async function loadSession(sessionId) {
  setStatus("Loading session...");
  const session = await fetchJson(`/sessions/${sessionId}`);
  state.activeSessionId = sessionId;
  state.activeSession = session;
  renderActiveSession();
  setStatus("");
}

async function ensureSession() {
  await refreshSessions();
  if (state.sessions.length === 0) {
    const created = await fetchJson("/sessions", { method: "POST" });
    state.sessions = [created];
  }
  await loadSession(state.sessions[0].session_id);
}

async function createSession() {
  setStatus("Creating session...");
  const created = await fetchJson("/sessions", { method: "POST" });
  await refreshSessions();
  await loadSession(created.session_id);
  inputEl.focus();
}

async function sendMessage(event) {
  event.preventDefault();
  if (!state.activeSessionId) {
    return;
  }

  const message = inputEl.value.trim();
  if (!message) {
    return;
  }

  const includeTiming = timingEl.checked;
  formEl.querySelector("button").disabled = true;
  setStatus("Generating answer...");

  try {
    const response = await fetchJson(`/sessions/${state.activeSessionId}/chat`, {
      method: "POST",
      body: JSON.stringify({ message, include_timing: includeTiming }),
    });
    await refreshSessions();
    await loadSession(state.activeSessionId);
    const assistantMessages = state.activeSession.messages.filter((item) => item.role === "assistant");
    const latestAssistant = assistantMessages[assistantMessages.length - 1];
    if (latestAssistant) {
      state.responsePayloads.set(responseStorageId(state.activeSessionId, latestAssistant.created_at), response);
      saveResponsePayloads();
    }
    renderActiveSession();
    inputEl.value = "";
    setStatus("");
  } catch (error) {
    setStatus(error.message || "Failed to send message.", true);
  } finally {
    formEl.querySelector("button").disabled = false;
  }
}

newSessionButtonEl.addEventListener("click", createSession);
formEl.addEventListener("submit", sendMessage);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.metaKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

loadResponsePayloads();
ensureSession().catch((error) => {
  setStatus(error.message || "Failed to load app.", true);
});
