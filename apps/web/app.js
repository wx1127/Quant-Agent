const $ = (id) => document.getElementById(id);
const token = () => localStorage.getItem('quantAgentToken') || 'web:research';
const api = () => $('apiBase').value.replace(/\/$/, '');
async function get(path) {
  const response = await fetch(`${api()}${path}`, {headers: {Authorization: `Bearer ${token()}`} });
  if (!response.ok) throw new Error(`${response.status} ${path}`);
  return response.json();
}
function renderSourceStatus(source) {
  $('sourceState').textContent = source.research_provider_connected ? '已装配' : '未装配';
  const tokenState = source.token_present ? 'Token 已配置' : '未配置 Token';
  const ttl = source.research_cache_ttl_seconds;
  $('sourceMeta').textContent = `${tokenState} · 缓存 ${ttl}s`;
}
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char])); }
function meta(result) { const freshness = result.freshness === 'cached' ? '缓存' : '实时'; return `${freshness} · ${result.as_of} · ${result.data_version} · ${result.model_version}`; }
function rows(target, items) { $(target).innerHTML = items.length ? items.map(item => `<div class="row"><b>${escapeHtml(item.symbol || item.name || item.industry || '—')}</b><em>${escapeHtml(item.score ?? item.state ?? item.rank ?? '')}</em></div>`).join('') : '<div class="status">暂无数据</div>'; }
async function refresh() {
  $('status').textContent = '正在读取研究快照…';
  try {
    const source = await get('/data/tushare-status');
    renderSourceStatus(source);
    const [market, mainlines, leaders, candidates] = await Promise.all([get('/research/market'), get('/research/mainlines'), get('/research/leaders?limit=8'), get('/research/candidates?limit=8')]);
    $('market').textContent = market.items[0]?.state || market.items[0]?.regime || '已加载'; $('marketMeta').textContent = meta(market);
    $('mainlineCount').textContent = mainlines.items.length; $('leaderCount').textContent = leaders.items.length; $('candidateCount').textContent = candidates.items.length;
    $('mainlineMeta').textContent = meta(mainlines); $('leadersMeta').textContent = meta(leaders); $('candidatesMeta').textContent = meta(candidates);
    rows('mainlines', mainlines.items); rows('leaders', leaders.items); rows('candidates', candidates.items);
    $('status').textContent = '研究快照加载完成';
  } catch (error) { try { const source = await get('/data/tushare-status'); renderSourceStatus(source); $('status').textContent = `暂无研究快照：${source.message}。请先执行数据同步。`; } catch (_) { $('status').textContent = `加载失败：${error.message}。请确认 API 地址、认证和数据快照。`; } }
}
$('refresh').addEventListener('click', refresh); refresh();
