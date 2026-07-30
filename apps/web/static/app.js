let bearerToken = "";

function bindToken() {
  const input = document.querySelector("#api-token");
  const button = document.querySelector("#save-token");
  if (!input || !button) return;
  button.addEventListener("click", () => {
    bearerToken = input.value.trim();
    input.value = "";
    setStatus(bearerToken ? "本次页面会话已加载访问令牌。" : "访问令牌为空。");
    loadPageData();
  });
}

async function api(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  if (bearerToken) headers.Authorization = `Bearer ${bearerToken}`;
  const response = await fetch(`/v1${path}`, {...options, headers});
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body?.error?.message || `请求失败：${response.status}`);
  }
  return body.data;
}

function setStatus(message) {
  const target = document.querySelector("#status");
  if (target) target.textContent = message;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function evidenceBlock(item) {
  const support = item.support_evidence || item.support || [];
  const counter = item.counter_evidence || item.counter || [];
  return `<div class="evidence">
    <div class="support"><strong>支持证据</strong><p>${escapeHtml(support.join("；") || "证据不足")}</p></div>
    <div class="counter"><strong>反对证据</strong><p>${escapeHtml(counter.join("；") || "证据不足")}</p></div>
  </div>`;
}

async function loadResearch() {
  const [regime, themes, candidates] = await Promise.all([
    api("/market/regime"),
    api("/themes?limit=5"),
    api("/candidates?limit=8"),
  ]);
  document.querySelector("#as-of").textContent = `数据截止：${regime.as_of}`;
  document.querySelector("#regime-state").textContent = regime.regime || regime.state || "暂无";
  document.querySelector("#regime-score").textContent = regime.score ?? "—";
  document.querySelector("#regime-evidence").innerHTML = evidenceBlock(regime);
  document.querySelector("#themes-body").innerHTML = themes.items.map((item) =>
    `<tr><td>${escapeHtml(item.name || item.id)}</td><td>${escapeHtml(item.state)}</td>
    <td>${escapeHtml(item.score)}</td><td>${escapeHtml(item.persistence_days)}</td></tr>`
  ).join("");
  document.querySelector("#candidates-body").innerHTML = candidates.items.map((item) =>
    `<tr><td><a href="/web/evidence?instrument_id=${encodeURIComponent(item.id)}">${escapeHtml(item.name || item.id)}</a></td><td>${escapeHtml(item.theme_id)}</td>
    <td>${escapeHtml(item.tier)}</td><td>${escapeHtml(item.score)}</td>
    <td>${item.tradable ? "可交易" : "仅观察"}</td></tr>`
  ).join("");
}

async function loadTrading() {
  setStatus("请输入账户范围内的草案 ID，再查看审批详情。");
}

async function loadPortfolio() {
  const accountId = document.querySelector("#account-id").value.trim();
  if (!accountId) return setStatus("请填写账户 ID。");
  try {
    const portfolio = await api(`/portfolios/${encodeURIComponent(accountId)}`);
    document.querySelector("#portfolio-detail").innerHTML = `
      <p>账户：${escapeHtml(portfolio.account_id)}</p>
      <p>数据截止：${escapeHtml(portfolio.as_of)}</p>
      <p>数据版本：${escapeHtml(portfolio.data_version)}</p>
      <p>总权益：${escapeHtml(portfolio.total_equity)}</p>`;
  } catch (error) {
    setStatus(error.message);
  }
}

async function startBacktest() {
  const payload = {
    strategy_id: document.querySelector("#strategy-id").value.trim(),
    parameter_version: document.querySelector("#parameter-version").value.trim(),
    data_version: document.querySelector("#backtest-data-version").value.trim(),
  };
  try {
    const task = await api("/backtests", {method: "POST", body: JSON.stringify(payload)});
    document.querySelector("#backtest-result").textContent =
      `任务 ${task.run_id} 已进入 ${task.status}，可通过 API 查询后续结果。`;
  } catch (error) {
    setStatus(error.message);
  }
}

async function loadEvidence() {
  const params = new URLSearchParams(window.location.search);
  const field = document.querySelector("#instrument-id");
  if (params.get("instrument_id")) field.value = params.get("instrument_id");
  const instrumentId = field.value.trim();
  if (!instrumentId) return setStatus("请填写证券 ID。");
  try {
    const item = await api(`/instruments/${encodeURIComponent(instrumentId)}/evidence`);
    document.querySelector("#evidence-title").textContent = item.name || item.id;
    document.querySelector("#evidence-as-of").textContent = `数据截止：${item.as_of} · ${item.data_version}`;
    document.querySelector("#evidence-detail").innerHTML = evidenceBlock(item) +
      `<p><strong>失效条件</strong>：${escapeHtml((item.invalidations || []).join("；") || "证据不足")}</p>`;
  } catch (error) {
    setStatus(error.message);
  }
}

async function loadDraft() {
  const draftId = document.querySelector("#draft-id").value.trim();
  if (!draftId) return setStatus("请填写订单草案 ID。");
  try {
    const draft = await api(`/order-drafts/${encodeURIComponent(draftId)}`);
    document.querySelector("#draft-detail").innerHTML = `
      <p>决策 ID：${escapeHtml(draft.decision_id)}</p>
      <p>订单哈希：${escapeHtml(draft.batch_hash)}</p>
      <p>有效期：${escapeHtml(draft.expires_at)}</p>
      <p>预估费用：${escapeHtml(draft.estimated_fees)}</p>
      <div class="table-wrap"><table><thead><tr><th>证券</th><th>方向</th><th>数量</th>
      <th>参考价</th></tr></thead><tbody>${draft.orders.map((order) =>
        `<tr><td>${escapeHtml(order.instrument_id)}</td><td>${escapeHtml(order.side)}</td>
        <td>${escapeHtml(order.quantity)}</td><td>${escapeHtml(order.reference_price)}</td></tr>`
      ).join("")}</tbody></table></div>`;
    document.querySelector("#risk-list").innerHTML =
      draft.risk_warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("") ||
      "<li>当前草案没有附加警告，但仍需人工核对订单与账户。</li>";
    document.querySelector("#approve").disabled = false;
    document.querySelector("#approve").dataset.draftId = draftId;
  } catch (error) {
    setStatus(error.message);
  }
}

async function approveDraft(button) {
  const draftId = button.dataset.draftId;
  button.disabled = true;
  try {
    const approval = await api(`/order-drafts/${encodeURIComponent(draftId)}/approve`, {
      method: "POST",
      body: "{}",
    });
    document.querySelector("#approval-token").value = approval.approval_token;
    document.querySelector("#submit").disabled = false;
    document.querySelector("#submit").dataset.draftId = draftId;
    setStatus("审批已签发。提交前请再次核对风险和有效期。");
  } catch (error) {
    setStatus(error.message);
    button.disabled = false;
  }
}

async function submitDraft(button) {
  if (!button.dataset.idempotencyKey) {
    button.dataset.idempotencyKey = crypto.randomUUID();
  }
  const key = button.dataset.idempotencyKey;
  button.disabled = true;
  try {
    const result = await api(`/order-drafts/${encodeURIComponent(button.dataset.draftId)}/submit`, {
      method: "POST",
      headers: {"Idempotency-Key": key},
      body: JSON.stringify({approval_token: document.querySelector("#approval-token").value}),
    });
    setStatus(`模拟提交状态：${result.status || "已受理"}`);
  } catch (error) {
    setStatus(error.message);
    button.disabled = false;
  }
}

async function sendMessage() {
  const input = document.querySelector("#message");
  const message = input.value.trim();
  if (!message) return;
  appendBubble(message, "user");
  input.value = "";
  try {
    const answer = await api("/agent/messages", {
      method: "POST",
      body: JSON.stringify({message, conversation_id: "web-session"}),
    });
    const link = answer.decision_url
      ? `<a href="${escapeHtml(answer.decision_url)}">查看决策详情</a>`
      : "没有生成决策详情";
    appendAgentBubble(
      answer.summary,
      answer.agent_state,
      answer.decision_id,
      answer.as_of,
      answer.tool_references,
      link,
    );
  } catch (error) {
    appendBubble(`任务失败：${error.message}`, "agent");
  }
}

function appendAgentBubble(summary, state, decisionId, asOf, references, link) {
  const bubble = document.createElement("div");
  bubble.className = "bubble agent";
  bubble.innerHTML = `<p>${escapeHtml(summary)}</p>
    <p>状态：${escapeHtml(state)} · 决策 ID：${escapeHtml(decisionId || "未生成")}</p>
    <p>数据截止：${escapeHtml(asOf || "无")}</p>
    <p>工具引用：${escapeHtml((references || []).join(", ") || "无")}</p>
    <p>${link}</p>`;
  document.querySelector("#chat-log").appendChild(bubble);
}

function appendBubble(message, kind) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${kind}`;
  bubble.textContent = message;
  document.querySelector("#chat-log").appendChild(bubble);
}

function loadPageData() {
  const page = document.body.dataset.page;
  if (!bearerToken) return;
  const loader =
    page === "research" ? loadResearch :
    page === "trading" ? loadTrading :
    page === "evidence" ? loadEvidence : null;
  if (loader) loader().catch((error) => setStatus(error.message));
}

document.addEventListener("DOMContentLoaded", () => {
  bindToken();
  document.querySelector("#load-draft")?.addEventListener("click", loadDraft);
  document.querySelector("#load-portfolio")?.addEventListener("click", loadPortfolio);
  document.querySelector("#run-backtest")?.addEventListener("click", startBacktest);
  document.querySelector("#load-evidence")?.addEventListener("click", loadEvidence);
  document.querySelector("#approve")?.addEventListener("click", (event) => approveDraft(event.currentTarget));
  document.querySelector("#submit")?.addEventListener("click", (event) => submitDraft(event.currentTarget));
  document.querySelector("#send")?.addEventListener("click", sendMessage);
});
