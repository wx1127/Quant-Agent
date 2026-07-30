# 量化 Agent 项目开发文档

> 文档状态：Draft v0.1  
> 依赖文档：  
> - [01-agent-harness.md](./01-agent-harness.md)  
> - [02-project-overview.md](./02-project-overview.md)

## 1. 开发目标

构建一个支持中国内地股票和场内 ETF 的日频量化 Agent，完成：

- 数据采集、清洗、版本化和质量控制；
- 市场阶段、主线、龙头和候选股识别；
- ETF 轮动和股票增强策略；
- 回测、滚动验证和模拟交易；
- 组合构建、独立风控和订单草案；
- 人工审批、执行网关和成交核对；
- Agent 工具调用、解释、报告和审计；
- Web 控制台及对话入口。

## 2. 技术选型

第一版推荐：

| 层级 | 技术 |
|---|---|
| 语言 | Python 3.12 |
| API | FastAPI + Pydantic |
| 业务数据库 | PostgreSQL |
| 研究存储 | Parquet + DuckDB |
| 缓存/任务锁 | Redis，可在最小版本后接入 |
| 定时任务 | APScheduler；规模扩大后使用任务队列 |
| 数据计算 | Polars 或 Pandas + NumPy |
| 统计/模型 | SciPy、statsmodels、scikit-learn |
| 回测 | 自研事件层 + 向量化研究层 |
| 前端 | React + TypeScript |
| 图表 | ECharts |
| 容器 | Docker Compose |
| 测试 | pytest、Hypothesis、Playwright |
| 监控 | OpenTelemetry + Prometheus + Grafana |

技术选择原则：

- 先保证时点正确、可复现和交易规则准确；
- 第一版避免引入过多分布式组件；
- 研究计算与实盘执行进程隔离；
- 生产数据库迁移必须版本化。

## 3. 推荐仓库结构

```text
quant-agent/
├─ apps/
│  ├─ api/                   # FastAPI 服务
│  ├─ worker/                # 数据和研究任务
│  ├─ execution_gateway/     # 模拟盘/券商执行
│  └─ web/                   # Web 控制台
├─ packages/
│  ├─ core/                  # 时间、标识、通用模型
│  ├─ data/                  # 数据源、清洗、质量检查
│  ├─ features/              # 因子和特征
│  ├─ regime/                # 市场阶段
│  ├─ themes/                # 主线识别
│  ├─ leaders/               # 龙头与候选
│  ├─ strategies/            # ETF和股票策略
│  ├─ portfolio/             # 组合构建
│  ├─ backtest/              # 回测引擎
│  ├─ risk/                  # 独立风控
│  ├─ execution/             # 订单模型和适配器
│  ├─ reconciliation/        # 成交与持仓核对
│  ├─ agent/                 # Harness、工具和提示词
│  ├─ reports/               # 日报与归因
│  └─ observability/         # 日志、指标和追踪
├─ configs/
│  ├─ environments/
│  ├─ strategies/
│  ├─ risk/
│  └─ universes/
├─ data_contracts/
├─ migrations/
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ replay/
│  ├─ adversarial/
│  └─ e2e/
├─ notebooks/                # 仅研究，不作为生产逻辑来源
├─ docs/
└─ docker-compose.yml
```

生产策略逻辑不得只存在于 Notebook 中。验证通过后必须迁移到带测试的包。

## 4. 领域模型

### 4.1 核心实体

| 实体 | 说明 |
|---|---|
| `Instrument` | 股票、ETF、指数等证券主数据 |
| `TradingCalendar` | 交易日和市场时区 |
| `MarketBar` | 未复权行情 |
| `AdjustmentFactor` | 复权因子 |
| `CorporateAction` | 分红、拆并股等公司行动 |
| `FundamentalPoint` | 带公告日和可用日的财务数据 |
| `IndustryMembership` | 带生效区间的行业归属 |
| `MarketRegime` | 市场阶段结果 |
| `ThemeScore` | 主线评分 |
| `LeaderScore` | 龙头评分 |
| `CandidateScore` | 候选股评分 |
| `StrategyRun` | 策略运行记录 |
| `DecisionSnapshot` | 不可变决策快照 |
| `PortfolioSnapshot` | 资金与持仓快照 |
| `OrderDraft` | 订单草案 |
| `Approval` | 人工审批 |
| `BrokerOrder` | 券商委托 |
| `Fill` | 成交记录 |
| `ReconciliationResult` | 核对结果 |
| `AuditEvent` | 审计事件 |

### 4.2 时间字段

所有市场相关表至少区分：

- `event_time`：事件实际发生时间；
- `available_at`：系统可合法使用该数据的最早时间；
- `ingested_at`：进入系统的时间；
- `version`：数据版本；
- `source`：来源。

回测查询必须满足：

```text
available_at <= decision_time
```

这是防止未来数据泄漏的基础条件。

## 5. 数据层

### 5.1 数据分层

```text
RAW       原始响应，不修改
STAGED    结构化、字段统一
CURATED   去重、校验、可研究
FEATURE   因子和模型特征
SNAPSHOT  决策与回测冻结版本
```

### 5.2 核心数据表

建议至少包含：

```text
instrument
trading_calendar
market_bar_daily
adjustment_factor
instrument_status
corporate_action
financial_statement_point_in_time
industry_classification
industry_membership_history
fund_nav
index_constituent_history
announcement
event_evidence
data_quality_result
dataset_version
```

### 5.3 数据质量规则

- 证券代码可解析且在生效期内；
- 交易日完整；
- OHLC 关系合法；
- 成交量和成交额非负；
- 无无法解释的重复记录；
- 复权因子连续性合理；
- 停复牌与行情状态一致；
- 行业归属具有生效日期；
- 财务数据具有公告日和可用时间；
- 关键行情与备用来源偏差不超过阈值；
- 数据异常时不得静默以前值填充。

### 5.4 数据源适配器

定义统一协议：

```python
class MarketDataProvider(Protocol):
    def fetch_instruments(self, as_of: datetime) -> list[Instrument]: ...
    def fetch_daily_bars(
        self, trade_date: date, instrument_ids: list[str]
    ) -> list[MarketBar]: ...
    def fetch_status(
        self, trade_date: date, instrument_ids: list[str]
    ) -> list[InstrumentStatus]: ...
```

适配器负责外部字段映射，领域层不得依赖某个数据源的原始字段名。

## 6. 特征工程

### 6.1 市场特征

- 指数 20/60/120日趋势；
- 全市场位于20/60日均线上方的比例；
- 上涨/下跌股票数量；
- 20/60日新高与新低比例；
- 成交额20日分位数；
- 个股收益率中位数；
- 行业上涨覆盖率；
- 大小盘相对强弱；
- 实现波动率和下行波动率；
- 涨停、跌停和炸板统计。

### 6.2 主线特征

- 5/20/60日行业相对收益；
- 板块上涨覆盖率；
- 板块成交额相对20日均值；
- 创新高股票占比；
- 龙头分数及持续性；
- 市场下跌日的相对抗跌性；
- 事件证据强度；
- 拥挤度和短期反转风险。

### 6.3 个股特征

- 相对所属行业和基准的强度；
- 均线斜率与趋势一致性；
- 突破及突破保持时间；
- 成交额、换手率和流动性分位数；
- 回调幅度和缩量程度；
- 新高距离；
- 基本面质量；
- 公告和风险事件；
- 停牌、涨跌停和最小成交金额。

所有滚动特征必须明确：

- 窗口；
- 最少观测数；
- 缺失处理；
- 是否复权；
- 计算时点；
- 版本。

## 7. 市场阶段引擎

### 7.1 基准规则

初始规则：

```text
regime_score =
0.25 × trend_score
+ 0.20 × breadth_score
+ 0.15 × turnover_score
+ 0.15 × new_high_low_score
+ 0.15 × diffusion_score
+ 0.10 × risk_score
```

所有子分标准化至0～100。

状态映射由配置控制，初始可设：

```text
>= 70          UPTREND
55 to < 70     RANGE_STRONG
40 to < 55     DIVERGENT
< 40           DOWNTREND
```

`BOTTOM_RECOVERY` 不能只靠总分判断，需要额外满足：

- 前期存在显著下跌；
- 新低比例持续下降；
- 市场宽度开始修复；
- 指数或中位数收益企稳；
- 状态持续满足确认天数。

### 7.2 防抖

为避免状态频繁切换：

- 使用确认天数；
- 使用进入/退出不同阈值；
- 记录原始分和最终状态；
- 禁止使用未来数据进行平滑。

### 7.3 输出

```python
class MarketRegimeResult(BaseModel):
    as_of: datetime
    regime: Regime
    raw_score: float
    confidence: float
    max_risk_budget: float
    evidence: list[Evidence]
    counter_evidence: list[Evidence]
    invalidations: list[str]
    model_version: str
```

## 8. 主线识别引擎

### 8.1 基准评分

```text
theme_score =
0.25 × relative_strength
+ 0.20 × breadth
+ 0.15 × turnover_expansion
+ 0.15 × leader_persistence
+ 0.10 × new_high_ratio
+ 0.10 × event_support
+ 0.05 × downside_resilience
- crowding_penalty
```

### 8.2 确认规则

`CONFIRMED` 主线必须满足：

- 当前排名进入前3；
- 最近5个交易日至少3天进入前5；
- 上涨覆盖率达到配置要求；
- 成交额增量不是单只股票贡献；
- 至少存在一个通过流动性过滤的龙头；
- 没有重大反向证据。

### 8.3 事件证据

事件模型只负责提供辅助证据：

```python
class EventEvidence(BaseModel):
    source_type: Literal["exchange", "company", "government", "licensed_news"]
    source_url: str
    published_at: datetime
    entities: list[str]
    event_type: str
    direction: Literal["positive", "negative", "neutral"]
    confidence: float
    excerpt_hash: str
```

事件证据不能单独把某板块升级为已确认主线。

## 9. 龙头与候选引擎

### 9.1 龙头评分

```text
leader_score =
0.25 × within_theme_strength
+ 0.15 × trend_quality
+ 0.15 × liquidity
+ 0.15 × theme_leadership
+ 0.10 × downside_resilience
+ 0.10 × logic_relevance
+ 0.10 × fundamental_quality
- risk_penalty
```

### 9.2 候选评分

```text
candidate_score =
0.20 × market_regime
+ 0.20 × theme_score
+ 0.20 × leader_score
+ 0.15 × price_trend
+ 0.10 × volume_price_structure
+ 0.10 × fundamental_or_event
+ 0.05 × valuation
- risk_penalty
```

权重配置需要版本化，研究完成后冻结。

### 9.3 可交易性过滤

按以下顺序过滤：

1. 证券状态；
2. 风险标记与退市状态；
3. 上市天数；
4. 停牌；
5. 涨跌停和预计可成交性；
6. 20日成交额和资金容量；
7. 账户或策略白名单；
8. 重大风险公告；
9. 组合集中度。

过滤失败的股票可以保留在研究报告中，但不得进入订单草案。

### 9.4 标签与概率校准

研究标签示例：

```text
label_5d  = 未来5日个股收益 - 同期基准收益
label_10d = 未来10日个股收益 - 同期行业收益
label_20d = 未来20日个股收益 - 同期基准收益
```

概率输出必须经过样本外校准。未校准前只显示分数和排名，不显示“上涨概率”。

## 10. ETF 策略

### 10.1 标的池

通过配置维护：

- 宽基 ETF；
- 风格 ETF；
- 行业 ETF；
- 黄金 ETF；
- 债券或现金管理类产品。

纳入条件包括上市时间、成交额、规模、跟踪误差和产品状态。

### 10.2 基准信号

```text
momentum =
0.5 × return_20d
+ 0.3 × return_60d
+ 0.2 × return_120d

risk_adjusted_score = momentum / volatility_60d
```

再执行：

- 中期趋势过滤；
- 选择前1～3只；
- 波动率或风险预算分配；
- 无合格标的时降低权益仓位；
- 周度检查、月度或阈值触发调仓。

参数只作为起始研究配置。

## 11. 回测引擎

### 11.1 必须模拟

- 当时可知的数据；
- 复权与公司行动；
- 交易日历；
- T+1 等适用交易规则；
- 最小交易单位；
- 手续费及最低收费；
- 印花税等适用费用；
- 滑点；
- 停牌；
- 涨跌停与不可成交；
- 部分成交；
- 组合资金约束。

费用规则必须按市场和生效日期版本化，不能永久硬编码为一个值。

### 11.2 回测模式

- 向量化研究：用于快速因子筛选；
- 事件驱动回测：用于最终策略验证；
- 历史回放：用于完整 Agent 决策验证。

### 11.3 数据划分

- 训练期；
- 验证期；
- 冻结样本外期；
- 滚动前推；
- 模拟盘。

时间序列数据不得随机打乱后划分。

### 11.4 评估指标

- 年化收益与基准超额；
- 最大回撤；
- 夏普、Sortino；
- 换手率和成本占比；
- 月度胜率；
- 信息系数和分层收益；
- 主线 Precision@K；
- 候选未来5/10/20日表现；
- 最大不利波动；
- 参数稳定性；
- 市场状态分组表现；
- 容量与不可成交比例。

## 12. 组合与风控

### 12.1 组合服务输入

```python
class PortfolioRequest(BaseModel):
    decision_id: str
    account_snapshot_id: str
    regime_result_id: str
    candidate_result_ids: list[str]
    strategy_version: str
    parameter_version: str
```

### 12.2 组合服务输出

- 当前权重；
- 目标权重；
- 目标与当前差异；
- 预计换手；
- 预计成本；
- 行业和风格暴露；
- 风险贡献；
- 生成原因。

### 12.3 风控服务

风控为独立包/服务，不能引用 Agent 结论作为唯一依据。接口：

```python
class RiskDecision(BaseModel):
    passed: bool
    violations: list[RiskViolation]
    warnings: list[RiskWarning]
    checked_policy_version: str
    checked_at: datetime
```

只要存在硬违规，订单草案不得进入审批。

## 13. 订单、审批和执行

### 13.1 订单生命周期

```text
DRAFT
→ RISK_REJECTED
→ PENDING_APPROVAL
→ APPROVED
→ SUBMITTING
→ ACCEPTED
→ PARTIALLY_FILLED
→ FILLED
→ CANCELED / REJECTED / EXPIRED
→ RECONCILED / INCIDENT
```

### 13.2 幂等键

推荐：

```text
idempotency_key =
hash(account_id, decision_id, instrument_id, side, quantity, order_batch_version)
```

执行网关提交前先检查本地订单映射，接口超时后先查询券商状态，禁止盲目重试。

### 13.3 审批哈希

审批绑定：

```text
order_batch_hash =
hash(sorted_orders, account_id, decision_id, expires_at)
```

价格、数量、方向或账户变化均使原审批失效。

### 13.4 券商适配器

```python
class BrokerAdapter(Protocol):
    def get_account_snapshot(self) -> AccountSnapshot: ...
    def submit_order(self, order: ApprovedOrder) -> BrokerOrder: ...
    def get_order(self, broker_order_id: str) -> BrokerOrder: ...
    def cancel_order(self, broker_order_id: str) -> BrokerOrder: ...
    def list_fills(self, since: datetime) -> list[Fill]: ...
```

模拟盘和真实券商实现同一协议，但使用不同凭证、进程和数据库。

## 14. Agent Harness 实现

### 14.1 Agent 工具

第一版注册：

```text
get_market_snapshot
validate_market_data
detect_market_regime
rank_market_themes
rank_theme_leaders
rank_stock_candidates
explain_candidate
run_backtest
get_portfolio_snapshot
build_target_portfolio
check_portfolio_risk
create_order_draft
get_order_draft
submit_approved_orders
reconcile_account
generate_daily_report
```

`approve_order_batch` 不注册为 Agent 工具，只允许审批页面调用。

### 14.2 编排规则

- 使用显式状态机，不依赖模型自由循环；
- 每个阶段限制可调用工具集合；
- 设置最大工具调用次数和总超时；
- 写操作前重新读取权威账户状态；
- 工具错误原样进入状态，不让模型猜测恢复；
- 所有回答引用 `decision_id` 和 `as_of`。

### 14.3 结构化输出

Agent 最终输出模型：

```python
class AgentAnswer(BaseModel):
    decision_id: str | None
    as_of: datetime | None
    answer_type: str
    summary: str
    facts: list[Fact]
    inferences: list[Inference]
    counter_evidence: list[Evidence]
    risks: list[RiskWarning]
    actions: list[AllowedAction]
    data_versions: list[str]
```

### 14.4 P6 已实现模块

0.7.0 的实现位于 `packages/quant_agent/agent/`：

- `snapshots.py` 固定一次决策使用的全部权威版本；
- `tools/registry.py` 实现未注册即拒绝、模式/状态权限、参数校验和审计；
- `tools/adapters.py` 将研究、组合、风控和模拟执行封装为标准工具响应；
- `runtime.py` 强制状态转换、调用次数、超时和取消；
- `responses.py` 要求数字引用具体工具响应字段；
- `security.py` 阻止外部文本或用户指令提升权限；
- `evaluation.py` 与 `audit_replay.py` 提供黄金轨迹评测和决策回放。

工具适配器只负责边界和编排，不复制底层市场、策略、组合或风控计算。新增工具必须
同时声明类别、读写属性、允许模式、允许状态、参数模型和服务版本。

## 15. API 设计

建议 API：

```text
GET  /v1/market/regime
GET  /v1/themes
GET  /v1/themes/{theme_id}/leaders
GET  /v1/candidates
GET  /v1/instruments/{instrument_id}/evidence

POST /v1/backtests
GET  /v1/backtests/{run_id}

GET  /v1/portfolios/{account_id}
POST /v1/portfolio-proposals
POST /v1/risk/checks

POST /v1/order-drafts
GET  /v1/order-drafts/{draft_id}
POST /v1/order-drafts/{draft_id}/approve
POST /v1/order-drafts/{draft_id}/submit
POST /v1/accounts/{account_id}/reconcile

POST /v1/agent/messages
GET  /v1/decisions/{decision_id}
GET  /v1/audit/events
POST /v1/runtime/kill-switch
```

所有 API 使用版本号、请求 ID 和统一错误结构。

### 15.1 P7 已实现模块

0.8.0 的应用入口为 `apps.api.app:app`：

- `apps/api/core/`：Bearer 认证、角色/账户授权、请求 ID、错误脱敏和服务边界；
- `apps/api/routes/research.py`：市场、主线、龙头、候选和个股证据；
- `apps/api/routes/backtest_portfolio.py`：异步回测、账户快照、独立风控和组合提案；
- `apps/api/routes/orders.py`：人工审批、一次性令牌、幂等模拟提交和核对；
- `apps/api/routes/agent.py`：不能切换模式或代替审批的研究对话；
- `packages/quant_agent/reports/daily.py`：结构化日报及 Markdown/HTML 渲染；
- `apps/web/`：研究、证据、回测、组合、风险、审批和对话页面。

组合提案的风控结果必须由服务端 `risk_handler` 产生，客户端不能提交 `passed=true`
绕过风控。审批令牌最长五分钟有效，绑定审批人、账户、决策 ID、草案和订单哈希，
修改草案、令牌过期、重复消费或 Kill Switch 激活都会拒绝提交。

## 16. 任务调度

### 16.1 收盘后任务

```text
更新交易日状态
→ 采集日线和证券状态
→ 数据质量检查
→ 生成冻结数据版本
→ 计算特征
→ 市场阶段
→ 主线
→ 龙头与候选
→ 策略和组合
→ 风控
→ 模拟盘
→ 日报
```

只有上游成功且版本一致，下游任务才可运行。

### 16.2 开盘前任务

- 公告和停复牌检查；
- 隔夜状态更新；
- 订单草案再验证；
- 过期草案失效；
- 生成开盘前风险提示。

### 16.3 盘中任务

第一版仅执行：

- 订单状态查询；
- 成交核对；
- 风险告警；
- Kill Switch。

第一版不做盘中自由选股和自动追涨。

## 17. 安全与权限

角色建议：

| 角色 | 权限 |
|---|---|
| Viewer | 查看研究结果 |
| Researcher | 发起研究和回测 |
| Trader | 创建订单草案 |
| Approver | 批准特定账户订单 |
| RiskAdmin | 管理风控和 Kill Switch |
| SystemAdmin | 系统配置，不默认拥有交易审批权 |

要求：

- 生产密钥使用密钥管理服务；
- 日志脱敏；
- 实盘 API 网络访问受限；
- 审批使用强认证；
- 高风险操作记录不可变审计；
- 不在前端保存券商凭证；
- 研究环境不能访问实盘执行网络。

## 18. 测试策略

### 18.1 单元测试

- 收益和复权；
- 滚动窗口；
- 时点查询；
- 市场状态阈值和防抖；
- 主线持续性；
- 龙头和候选排序；
- 涨跌停、停牌、最小单位；
- 费用和滑点；
- 组合限制；
- 幂等键和订单状态机。

### 18.2 属性测试

示例：

- 仓位之和不得超过允许上限；
- 风险阈值收紧后可用风险预算不得增加；
- 使用更晚 `available_at` 的记录不会进入更早决策；
- 同一幂等键不会生成两个券商订单；
- 未审批订单永远不能进入提交状态。

### 18.3 集成测试

- 数据源到冻结版本；
- 冻结版本到日报；
- 策略到订单草案；
- 审批到模拟成交；
- 券商超时后的状态查询；
- 部分成交后的核对；
- Kill Switch 后拒绝新增订单。

### 18.4 Agent 对抗测试

- 用户要求忽略风控；
- 新闻正文包含伪造系统指令；
- 用户要求把模拟盘切换成实盘；
- 用户要求修改审批后的订单；
- 工具返回冲突数据；
- 用户询问“哪只股票一定上涨”；
- 用户要求模型直接编造缺失数据。

## 19. 环境与配置

环境划分：

```text
local
test
replay
paper
production
```

建议环境变量：

```text
APP_ENV
DATABASE_URL
RESEARCH_STORAGE_PATH
REDIS_URL
MARKET_DATA_PROVIDER
MARKET_DATA_TOKEN
LLM_PROVIDER
LLM_MODEL
LLM_API_KEY
BROKER_ADAPTER
BROKER_CREDENTIALS_REF
RUNTIME_MODE
KILL_SWITCH_DEFAULT
AUDIT_SIGNING_KEY_REF
```

配置规则：

- 密钥不进入仓库；
- 本地默认 `RESEARCH`；
- 测试默认使用伪造数据和模拟券商；
- `LIVE_AUTO` 在代码和配置两层禁用；
- 生产启动时校验风险配置和 Kill Switch 状态。

## 20. 部署

### 20.1 最小部署

第一版可使用：

- 一个 API 服务；
- 一个任务 Worker；
- 一个独立执行网关；
- PostgreSQL；
- Parquet/DuckDB 研究存储；
- Web 前端。

执行网关必须独立部署，即使其他模块初期采用单体架构。

### 20.2 发布流程

1. 静态检查和单元测试；
2. 集成和黄金回放；
3. 构建不可变镜像；
4. 部署测试环境；
5. 运行数据库迁移检查；
6. 影子运行；
7. 人工批准发布；
8. 生产健康检查；
9. 保留回退版本。

## 21. 开发阶段

### Sprint 0：工程基础，1周

- 仓库、代码规范和 CI；
- 配置、日志和错误结构；
- PostgreSQL 与研究存储；
- 领域模型和迁移。

### Sprint 1：数据底座，2周

- 日线、交易日历、证券状态；
- 原始和研究数据分层；
- 数据质量；
- 冻结数据版本。

### Sprint 2：市场与主线，2周

- 市场特征；
- 市场阶段；
- 行业相对强度；
- 主线持续性；
- 市场日报。

### Sprint 3：龙头与候选，2周

- 龙头特征；
- 可交易性过滤；
- 候选评分；
- 个股证据页；
- 历史分层评测。

### Sprint 4：策略和回测，2～3周

- ETF 轮动；
- 股票增强组合；
- 事件驱动交易规则；
- 滚动验证和报告。

### Sprint 5：Harness 与模拟盘，2～3周

- Agent 工具；
- 决策快照；
- 风控；
- 模拟账户；
- 订单状态机；
- 审计和黄金场景。

### Sprint 6：Web 与稳定性，2周

- 总览、主线、候选和回测页面；
- 对话入口；
- 风险中心；
- 监控和告警；
- 影子运行。

### Sprint 7：辅助实盘

- 券商适配器；
- 审批流程；
- 成交核对；
- Kill Switch 演练；
- 合规确认；
- 小资金灰度。

## 22. 编码规范

- 领域层不直接调用第三方 SDK；
- 时间必须带时区；
- 金额和数量避免使用二进制浮点误差；
- 公开函数使用类型标注；
- 所有策略参数来自版本化配置；
- 禁止在生产代码中使用无种子的随机行为；
- 数据查询显式传入 `as_of`；
- 所有写接口携带幂等键；
- 风控规则必须有对应测试；
- 错误不得被吞掉或转换为正常空结果。

## 23. Pull Request 完成定义

每个功能 PR 应满足：

- 需求和边界清晰；
- 类型检查通过；
- 单元测试和相关集成测试通过；
- 涉及时点的数据逻辑具有泄漏测试；
- 涉及交易的逻辑具有幂等和失败测试；
- 涉及风控的逻辑覆盖允许与拒绝路径；
- 数据库变更包含迁移和回滚说明；
- API 变更更新契约；
- 配置变更具有版本；
- 文档同步更新；
- 不包含密钥或真实账户数据。

## 24. v1 验收标准

### 研究

- 能按指定历史日期重建当时市场状态；
- 能输出主线、龙头、候选及证据；
- 能区分事实、评分和推断；
- 能对候选进行5/10/20日样本外分层评估。

### 回测

- 支持交易成本、停牌和涨跌停；
- 支持时间序列样本外和滚动验证；
- 每次结果绑定数据、策略和参数版本；
- 结果可以重复运行得到一致输出。

### Harness

- Agent 只能调用已注册工具；
- 数值不由模型计算或补造；
- 所有订单通过独立风控；
- 未审批订单无法提交；
- 审批变更后自动失效；
- 重复请求不产生重复订单；
- 账实不一致触发 Kill Switch。

### 运行

- 影子运行不少于20个交易日；
- 模拟盘连续运行不少于3个月；
- 日报成功率达到目标；
- 无高等级未解决缺陷；
- 自动实盘保持禁用。

## 25. 后续扩展

v1 稳定后再评估：

- 盘中市场阶段和主线变化；
- 更完整的公告事件图谱；
- 概念与产业链关系；
- 概率模型和在线校准；
- 多账户和多策略组合；
- 更精细的冲击成本模型；
- 港股或其他市场；
- 受限自动实盘。

任何扩展都不得绕过 Harness 的权限、快照、风控、审批和审计要求。
