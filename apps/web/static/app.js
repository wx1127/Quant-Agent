let bearerToken = "";
const sessionTokenKey = "quant-agent-api-token";

function rememberToken(token) {
  bearerToken = token;
  if (token) sessionStorage.setItem(sessionTokenKey, token);
  else sessionStorage.removeItem(sessionTokenKey);
}

function bindToken() {
  const input = document.querySelector("#api-token");
  const button = document.querySelector("#save-token");
  if (!input || !button) return;
  button.addEventListener("click", () => {
    rememberToken(input.value.trim());
    input.value = "";
    setStatus(bearerToken ? "本次页面会话已加载访问令牌。" : "访问令牌为空。");
    loadPageData();
  });
}

function loadTokenFromFragment() {
  const values = new URLSearchParams(window.location.hash.slice(1));
  const token = values.get("token")?.trim();
  if (token) {
    rememberToken(token);
    history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
    setStatus("本地只读研究会话已自动连接。");
    return true;
  }
  const remembered = sessionStorage.getItem(sessionTokenKey);
  if (!remembered) return false;
  bearerToken = remembered;
  setStatus("已恢复当前标签页的只读研究会话。");
  return true;
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

function stateLabel(value) {
  return ({
    UPTREND: "上升趋势",
    RANGE_STRONG: "强势震荡",
    DIVERGENT: "结构分化",
    DOWNTREND: "下降趋势",
    CONFIRMED: "已确认",
    WATCH: "观察中",
  })[value] || value || "暂无";
}

function roleLabel(value) {
  return value === "LEADER" ? "龙头个股" : "候选个股";
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${(number * 100).toFixed(2)}%`;
}

function formatTurnover(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (number >= 100000000) return `${(number / 100000000).toFixed(2)} 亿`;
  if (number >= 10000) return `${(number / 10000).toFixed(0)} 万`;
  return number.toFixed(0);
}

function formatPrice(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "—";
}

function formatVolume(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (number >= 100000000) return `${(number / 100000000).toFixed(2)} 亿股`;
  if (number >= 10000) return `${(number / 10000).toFixed(2)} 万股`;
  return `${number.toFixed(0)} 股`;
}

let lastKlineBars = [];

function drawKline(bars) {
  const canvas = document.querySelector("#kline-chart");
  if (!canvas || !bars?.length) return;
  lastKlineBars = bars;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  const padding = {top: 20, right: 58, bottom: 30, left: 16};
  const volumeHeight = 70;
  const priceBottom = height - padding.bottom - volumeHeight - 16;
  const highs = bars.map((item) => Number(item.high));
  const lows = bars.map((item) => Number(item.low));
  const maxPrice = Math.max(...highs);
  const minPrice = Math.min(...lows);
  const priceSpan = Math.max(maxPrice - minPrice, maxPrice * 0.01, 0.01);
  const plotWidth = width - padding.left - padding.right;
  const slot = plotWidth / bars.length;
  const candleWidth = Math.max(3, Math.min(12, slot * 0.58));
  const yPrice = (value) => padding.top + (maxPrice - value) / priceSpan * (priceBottom - padding.top);
  context.font = "11px system-ui, sans-serif";
  context.textAlign = "left";
  context.textBaseline = "middle";
  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + (priceBottom - padding.top) * index / 4;
    const label = maxPrice - priceSpan * index / 4;
    context.strokeStyle = "rgba(133, 151, 181, 0.16)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillStyle = "#8090aa";
    context.fillText(label.toFixed(2), width - padding.right + 7, y);
  }
  bars.forEach((bar, index) => {
    const x = padding.left + slot * (index + 0.5);
    const open = Number(bar.open);
    const close = Number(bar.close);
    const rising = close >= open;
    const color = rising ? "#f05b67" : "#39c98a";
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x, yPrice(Number(bar.high)));
    context.lineTo(x, yPrice(Number(bar.low)));
    context.stroke();
    const bodyTop = Math.min(yPrice(open), yPrice(close));
    const bodyHeight = Math.max(1.5, Math.abs(yPrice(open) - yPrice(close)));
    context.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
  });
  const maxVolume = Math.max(...bars.map((item) => Number(item.volume)), 1);
  const volumeTop = priceBottom + 16;
  bars.forEach((bar, index) => {
    const x = padding.left + slot * (index + 0.5);
    const volume = Number(bar.volume);
    const barHeight = volume / maxVolume * volumeHeight;
    context.fillStyle = Number(bar.close) >= Number(bar.open)
      ? "rgba(240, 91, 103, 0.48)" : "rgba(57, 201, 138, 0.48)";
    context.fillRect(x - candleWidth / 2, height - padding.bottom - barHeight, candleWidth, barHeight);
  });
  const labelIndexes = [...new Set([0, Math.floor((bars.length - 1) / 2), bars.length - 1])];
  context.fillStyle = "#8090aa";
  labelIndexes.forEach((index) => {
    const x = padding.left + slot * (index + 0.5);
    context.textAlign = index === 0 ? "left" : index === bars.length - 1 ? "right" : "center";
    context.fillText(bars[index].trade_date.slice(5), x, height - 11);
  });
}

function renderStockDetail(item) {
  document.querySelector("#evidence-title").textContent = item.name || item.id;
  document.querySelector("#stock-symbol").textContent = `${item.symbol || item.id} · ${item.theme_name || "未分类"}`;
  document.querySelector("#latest-price").textContent = formatPrice(item.latest_price);
  const priceChange = document.querySelector("#price-change");
  const changeClass = Number(item.change) >= 0 ? "positive" : "negative";
  priceChange.className = `price-change ${changeClass}`;
  priceChange.textContent = `${Number(item.change) >= 0 ? "+" : ""}${formatPrice(item.change)}  ${Number(item.change_pct) >= 0 ? "+" : ""}${formatPercent(item.change_pct)}`;
  document.querySelector("#stock-open").textContent = formatPrice(item.open);
  document.querySelector("#stock-high").textContent = formatPrice(item.high);
  document.querySelector("#stock-low").textContent = formatPrice(item.low);
  document.querySelector("#stock-previous-close").textContent = formatPrice(item.previous_close);
  document.querySelector("#stock-volume").textContent = formatVolume(item.volume);
  document.querySelector("#stock-turnover").textContent = formatTurnover(item.turnover);
  const bars = item.bars || [];
  document.querySelector("#kline-range").textContent = bars.length
    ? `${bars[0].trade_date} 至 ${bars[bars.length - 1].trade_date}` : "暂无行情";
  drawKline(bars);
}

async function loadResearch() {
  const [regime, themes, candidates] = await Promise.all([
    api("/market/regime"),
    api("/themes?limit=8"),
    api("/candidates?limit=20"),
  ]);
  const marketAsOf = regime.market_as_of || regime.as_of;
  document.querySelector("#as-of").textContent = `行情截止：${marketAsOf} · 研究快照：${regime.as_of}`;
  document.querySelector("#regime-state").textContent = stateLabel(regime.regime || regime.state);
  document.querySelector("#regime-score").textContent = regime.score ?? "—";
  const meter = document.querySelector("#regime-meter-fill");
  if (meter) meter.style.width = `${Math.max(0, Math.min(100, Number(regime.score) || 0))}%`;
  document.querySelector("#regime-evidence").innerHTML = evidenceBlock(regime);
  document.querySelector("#themes-grid").innerHTML = themes.items.map((item) =>
    `<a class="theme-tile" href="/web/theme?theme_id=${encodeURIComponent(item.id)}">
      <div class="theme-rank">#${escapeHtml(item.rank || "—")}</div>
      <div><span class="theme-kind">${item.kind === "INDUSTRY" ? "行业板块" : "研究板块"}</span>
      <h3>${escapeHtml(item.name || item.id)}</h3>
      <p>${escapeHtml(item.scored_member_count || item.member_count || 0)} 只评分股票 · ${stateLabel(item.state)}</p></div>
      <div class="theme-score"><strong>${escapeHtml(item.score)}</strong><small>综合分</small></div>
    </a>`
  ).join("");
  document.querySelector("#candidates-body").innerHTML = candidates.items.map((item) =>
    `<tr><td><a class="stock-link" href="/web/evidence?instrument_id=${encodeURIComponent(item.id)}"><strong>${escapeHtml(item.name || item.id)}</strong><small>${escapeHtml(item.symbol || item.id)}</small></a></td>
    <td><a href="/web/theme?theme_id=${encodeURIComponent(item.theme_id)}">${escapeHtml(item.theme_name || item.theme_id)}</a></td>
    <td><span class="role-tag ${item.role === "LEADER" ? "leader-tag" : ""}">${roleLabel(item.role)}</span></td>
    <td>${escapeHtml(item.tier)}</td><td><strong>${escapeHtml(item.score)}</strong></td>
    <td>${item.tradable ? "可交易" : "仅观察"}</td></tr>`
  ).join("");
}

async function loadTheme() {
  const themeId = new URLSearchParams(window.location.search).get("theme_id");
  if (!themeId) throw new Error("缺少板块 ID。");
  const [theme, stocks] = await Promise.all([
    api(`/themes/${encodeURIComponent(themeId)}`),
    api(`/themes/${encodeURIComponent(themeId)}/stocks?limit=500`),
  ]);
  document.querySelector("#theme-title").textContent = theme.name || theme.id;
  document.querySelector("#theme-as-of").textContent = `数据截止：${theme.as_of} · ${stocks.total} 只评分股票`;
  document.querySelector("#theme-summary").innerHTML = `
    <div><small>主线排名</small><strong>#${escapeHtml(theme.rank)}</strong></div>
    <div><small>行业综合分</small><strong>${escapeHtml(theme.score)}</strong></div>
    <div><small>状态</small><strong>${stateLabel(theme.state)}</strong></div>
    <div><small>5日中位收益</small><strong>${formatPercent(theme.median_return_5d)}</strong></div>
    <div><small>上涨家数占比</small><strong>${formatPercent(theme.advancing_ratio)}</strong></div>`;
  document.querySelector("#theme-stocks-body").innerHTML = stocks.items.map((item) =>
    `<tr><td><span class="rank-number ${Number(item.rank) <= 3 ? "top-rank" : ""}">${escapeHtml(item.rank)}</span></td>
    <td><a class="stock-link" href="/web/evidence?instrument_id=${encodeURIComponent(item.id)}"><strong>${escapeHtml(item.name || item.id)}</strong><small>${escapeHtml(item.symbol || item.id)}</small></a></td>
    <td><span class="role-tag ${item.role === "LEADER" ? "leader-tag" : ""}">${escapeHtml(item.leader_type || roleLabel(item.role))}</span></td>
    <td><strong>${escapeHtml(item.score)}</strong></td><td class="${Number(item.return_5d) >= 0 ? "positive" : "negative"}">${formatPercent(item.return_5d)}</td>
    <td class="${Number(item.return_20d) >= 0 ? "positive" : "negative"}">${formatPercent(item.return_20d)}</td><td>${formatTurnover(item.turnover)}</td></tr>`
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
    const [detail, evidence] = await Promise.all([
      api(`/instruments/${encodeURIComponent(instrumentId)}`),
      api(`/instruments/${encodeURIComponent(instrumentId)}/evidence`),
    ]);
    renderStockDetail(detail);
    document.querySelector("#evidence-as-of").textContent = `行情截止：${detail.trade_date} · 研究快照：${detail.as_of}`;
    document.querySelector("#evidence-detail").innerHTML = evidenceBlock(evidence) +
      `<p><strong>失效条件</strong>：${escapeHtml((evidence.invalidations || []).join("；") || "证据不足")}</p>`;
    setStatus("");
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
    page === "theme" ? loadTheme :
    page === "trading" ? loadTrading :
    page === "evidence" ? loadEvidence : null;
  if (loader) loader().catch((error) => setStatus(error.message));
}

document.addEventListener("DOMContentLoaded", () => {
  bindToken();
  const bootstrapped = loadTokenFromFragment();
  document.querySelector("#load-draft")?.addEventListener("click", loadDraft);
  document.querySelector("#load-portfolio")?.addEventListener("click", loadPortfolio);
  document.querySelector("#run-backtest")?.addEventListener("click", startBacktest);
  document.querySelector("#load-evidence")?.addEventListener("click", loadEvidence);
  document.querySelector("#approve")?.addEventListener("click", (event) => approveDraft(event.currentTarget));
  document.querySelector("#submit")?.addEventListener("click", (event) => submitDraft(event.currentTarget));
  document.querySelector("#send")?.addEventListener("click", sendMessage);
  document.querySelector("#back-button")?.addEventListener("click", () => {
    if (document.referrer) history.back();
    else window.location.href = "/web";
  });
  if (bootstrapped) loadPageData();
});

window.addEventListener("resize", () => {
  if (lastKlineBars.length) drawKline(lastKlineBars);
});
