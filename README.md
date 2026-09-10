# Quant Agent

Quant Agent 是一个面向中国内地股票与场内 ETF 的日频量化研究和受控交易辅助系统。

系统采用以下边界：

- 大模型负责理解、编排与解释；
- 确定性程序负责行情、因子、回测和组合计算；
- 独立风控拥有订单否决权；
- 第一版实盘订单必须人工审批；
- 第一版明确禁用自动实盘。

## 当前进度

P0～P5 的主要基础已经实现，P6 Agent Harness 已完成不可变决策快照、工具权限内核、七个只读
研究工具与七个组合/风控/模拟交易工具：工程基础、
数据底座、Point-in-time 研究、龙头候选、策略、回测、滚动验证、账户快照、跨策略目标组合、
独立组合风控、不可执行订单草案、模拟成交、成交与持仓核对、全局/账户 Kill Switch，以及绑定
数据、策略参数、风险政策、账户、代码和运行组件版本的决策身份。当前具备可恢复的 RAW-first
数据同步、五类市场状态、ETF 与股票目标策略、中国市场规则、成本前后回测、失败时期分析，
以及版本/哈希绑定的仓位、行业、换手、回撤检查、人工审批前草案、无券商权限的 PIT 模拟撮合、
独立差异证据、持久化停机控制、write-once 决策快照，以及按服务端运行模式固定的 Agent 工具
白名单、严格参数/输出验证、同步权限审计和决策绑定的研究、组合、风险、草案、模拟提交及核对
调用链，以及按目标/模式收敛工具、前置制品、调用预算、总超时、取消、人工审批和强制核对的
显式 Agent 状态机。当前 API、Web、报告、部署健康检查、回滚演练、Tushare 研究适配器和
Provider 异常脱敏均已有代码与测试。公告在线链路、P1.6 真实账户同步、P3 真实数据源验收、
远程 CI、券商联调和长期运行仍待真实环境验证；当前台账为 81/84 项任务完成，不能把当前版本
视为可自动实盘系统。详细状态见：

- [项目进度文档](./docs/04-project-progress.md)

设计与开发资料：

- [Agent Harness 设计文档](./docs/01-agent-harness.md)
- [项目说明文档](./docs/02-project-overview.md)
- [项目开发文档](./docs/03-development-guide.md)

## 快速开始

项目要求 Python 3.12。PowerShell 中创建隔离环境并安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
```

也可使用 Conda：

```powershell
conda env create -f environment.yml
conda activate quant-agent
```

运行确定性的本地演示流水线：

```powershell
.\.venv\Scripts\quant-agent.exe demo
# 等价入口：.\.venv\Scripts\python.exe -m quant_agent demo
```

该命令会使用内置股票与 ETF 样本，依次完成主数据和日历写入、幂等日线入库、
数据质量门禁、Parquet 快照、内容校验及 DuckDB 查询。重复执行时已有日线会被跳过，
快照内容哈希保持不变。命令输出不包含数据库凭证或 Provider Token。

运行确定性的 Point-in-time 研究演示：

```powershell
.\.venv\Scripts\quant-agent.exe research demo
```

输出包含市场趋势、宽度、五类市场状态、环境分、风险预算、行业排名和稳定结果哈希。
演示使用合成数据，只用于验证研究链路，不代表投资建议或历史业绩。

默认本地文件：

- SQLite：`artifacts/quant_agent.db`；
- 快照：`artifacts/snapshots/<data_version>/`。

## 配置与密钥

环境 TOML 位于 `configs/environments/`。`QuantAgentSettings.from_toml()` 只验证文件；
`QuantAgentSettings.load()` 还会应用白名单内的 `QUANT_AGENT_*` 环境变量覆盖。
运行时可解析 `env://NAME`；`vault://path` 必须显式注入 Vault resolver，否则失败关闭。

`.env.example` 是变量清单，项目不会隐式读取 `.env`。请通过进程环境或密钥管理器
注入真实值，不要把 Token、数据库密码或券商凭证写入 TOML、命令参数或日志。

例如指定另一个本地数据库和快照目录：

```powershell
$env:QUANT_AGENT_DATABASE_URL = 'sqlite:///./artifacts/custom.db'
$env:QUANT_AGENT_RESEARCH_STORAGE_PATH = './artifacts/custom-snapshots'
.\.venv\Scripts\quant-agent.exe demo
```

## 质量检查

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy packages
.\.venv\Scripts\python.exe -m pytest
```

最近一次本地完整门禁（2026-09-10）为 2107 项测试通过、总覆盖率 90.18%，同时通过 Ruff
格式/代码检查和 Mypy（171 个源文件）类型检查。该数字是当前工作树快照，不替代 CI 与真实
数据源验收。

## P5 模拟执行

模拟执行只接受 `PAPER` 账户快照和已通过风控的不可执行订单草案，不读取券商凭证，也不发起网络请求。
它使用版本化中国市场规则、费用、滑点和行情最大年龄完成一次确定性撮合，并按成交重放现金与 lot 持仓，
覆盖部分成交、无法成交冻结、T+1、奇数股全清和到期释放。`InMemoryPaperRepository` 用于测试，
`SQLitePaperRepository` 用于本地持久化；账户、回执、幂等键和批次索引在同一事务提交，进程重启后
重复请求仍返回原回执。账户刷新必须由链接上一快照哈希和事件日志的新快照驱动。

`PaperExecutionService` 必须注入 `new_order_gate`；服务显式核验允许决策，并在检查、撮合和提交
期间保持停机闸门锁。精确匹配的已提交回执可以重放；真正的新订单必须重新检查 GLOBAL 和 ACCOUNT 状态。

## P5 核对与停机

`ReconciliationEngine` 独立核对订单、逐笔成交、费用、现金及持仓，保留缺失、重复、冲突和
异常顺序回报，不会用净额抵消掩盖分项差异。核对结果绑定完整输入、决策、订单及停止信号；
小数计算与哈希不依赖调用方的全局 Decimal 精度。

`KillSwitchService` 支持全局和账户开关、事故累积、审批恢复及独立事件哈希链。
`SQLiteKillSwitchRepository` 在重启后保留状态、事故、操作和阻断决策，并通过 SQLite 写事务
让并发下单与开关变更串行完成。严重核对结果可通过 `activate_from_reconciliation()` 激活账户开关。

恢复默认拒绝。应用必须注入可信身份/审批适配器，验证人工风控管理员与完整审批记录；仓储还会在
取得锁后复核审批有效期和状态版本。数据库缺失、读取异常或数据损坏均不能默认放行。
应用装配、幂等行为和权限边界见 [Kill Switch 接入说明](./packages/quant_agent/risk/kill_switch/README.md)。

## P6 决策快照

`DecisionSnapshotService` 在决策开始时解析并校验实际已验证的数据清单、账户快照、可执行策略
配置、已登记参数和风险政策，再将它们与决策时点、运行模式、完整 Git commit、代码制品哈希、
Agent 版本及模型版本绑定。恢复已有决策时会重新解析当前输入；任一输入发生变化都必须创建新的
`decision_id`，不能把新数据静默装入旧决策。

`DecisionSnapshot` 使用严格、规范化的 JSON 和完整内容哈希保存上述身份。
`InMemoryDecisionSnapshotRepository` 与 `SQLiteDecisionSnapshotRepository` 均采用 write-once
语义：同一 ID 和完全相同内容可以幂等重放，同一 ID 的任何内容变化都会被拒绝。SQLite 实现
支持重启恢复、跨连接原子创建、严格 schema/JSON/哈希/行身份校验，并以数据库触发器阻断
`UPDATE`、`DELETE` 及同主键 `INSERT`/`REPLACE`；底表使用 `WITHOUT ROWID`，不留下可用于
替换另一身份的隐藏行键。

决策快照只建立输入身份和可重放边界，不授予下单或工具调用权限。文件完整性验证也不等于逐行
Point-in-time 正确性；后续确定性工具仍须按冻结的 `as_of` 过滤可用信息。组合/执行工具适配已由
P6-T04 交付，显式 Agent 状态机已由 P6-T05 交付；跨进程持久化的完整轨迹和制品聚合回放仍属于
后续任务。可信输入、存储及当前
支持范围详见
[决策快照接入说明](./packages/quant_agent/agent/snapshots/README.md)。

## P6 工具权限内核

`AgentToolRegistryBuilder` 只接受可信装配层显式注册、且位于固定 v1 中央策略中的工具；不会扫描
领域包、插件或任意 Python callable。注册器在构建时绑定服务端运行模式、principal、capability
和账户范围，模型参数不能覆盖这些权限上下文。`RESEARCH`、`BACKTEST`、`PAPER` 与
`LIVE_ASSISTED` 使用不同白名单，`LIVE_AUTO` 保持空集合并在构建时拒绝；人工审批、切换模式、
修改风控阈值和恢复 Kill Switch 均为不可注册保留入口。

参数和输出使用严格、冻结、拒绝额外字段的契约，原始 JSON 还会拒绝重复键、非有限数、过深或
过大的载荷。`submit_paper_orders` 只能在 `PAPER` 模式注册，
`submit_approved_orders` 只能在 `LIVE_ASSISTED` 模式注册；实盘提交不接受模型提供
审批令牌，而要求可信授权器返回一次性租约，并在允许审计和 handler 执行期间持续持有。
权限审计失败时执行失败关闭。七个只读研究 handler 与七个组合/风控/模拟交易 handler 已在此
内核上实现；当前尚未注册真实券商
执行 handler，也没有交付生产级审批服务或跨进程不可篡改审计。完整矩阵与装配边界见
[Agent 工具注册说明](./packages/quant_agent/agent/tools/README.md)。

## P6 研究分析工具

`ResearchToolset` 将市场快照、数据质量、市场阶段、市场主线、主题龙头、股票候选和候选解释封装
为七个只读 `AgentTool`。每次调用从已恢复的 `DecisionSnapshot` 取得 `decision_id`、`as_of`、
数据版本和内容哈希，并通过可信 `ResearchInputSource` 读取完整 PIT 输入；市场阶段与主线会重放
截至决策时点的完整历史状态机，最终数值直接来自现有领域引擎，不在 Agent 层复制评分公式。

所有输出均使用严格、冻结、拒绝额外字段的具体模型，并明确保持候选分数为“未校准评分”，不把
它包装成上涨概率。数据不可用、输入错位、质量门禁失败和候选不存在分别返回标准错误，不会被
转换为空排名。龙头、候选和解释虽然只读，但容量与可交易性依赖账户规模，因而必须同时匹配冻结
账户快照及注册器账户作用域。

当前没有提供覆盖全部领域对象的生产 Parquet/Arrow 解码 source，也不会自动注册合成或占位数据。
生产应用必须先实现并注入可信 `ResearchInputSource`，再在 composition root 显式注册这七个工具。
完整调用链、失败语义和接入边界见
[研究分析工具说明](./packages/quant_agent/agent/tools/research/README.md)。

## P6 组合、风控与模拟交易工具

`PortfolioExecutionToolset` 将冻结账户快照、目标组合、独立组合风控、不可执行订单草案、模拟
提交和账户核对封装为七个决策绑定工具。pipeline 在内部按固定顺序重建并校验前置制品：风险
`REJECT/ERROR` 不能落库草案，模拟提交只能读取账户/决策绑定的已存储批次，并且必须通过
`PaperExecutionService` 保留 Kill Switch 锁、仓储 CAS、批次唯一消费和持久化回执重放。

写工具要求账户作用域幂等键；并发重复请求由首次原子提交定义结果，后续调用只恢复相同草案或
回执。`submit_paper_orders` 仅在可信 composition 注入 gateway 后装配，中央权限只允许 `PAPER`；
`LIVE_ASSISTED` 不会获得模拟提交权限，也没有真实券商 handler。核对必须使用独立观察证据，返回
完整差异与停止信号但不在只读工具内修改 Kill Switch。

`InMemoryDraftArtifactStore` 只用于测试或单进程本地运行；生产应用仍须注入耐久事务仓储与权威
`PortfolioExecutionInputSource`。完整边界见
[组合执行工具说明](./packages/quant_agent/agent/tools/portfolio_execution/README.md)。

## P6 显式 Agent Runtime

`AgentRuntime` 是面向模型适配层的唯一工作流入口。它在中央 `AgentToolRegistry` 之上按服务端选择的
目标和当前状态收窄工具目录，并在真正分发前再次检查前置制品、订单批次和写操作幂等身份。研究、
回测、组合报告、订单草案、PAPER 执行和 LIVE_ASSISTED 执行使用不同的合法路径；风控、审批、提交
和核对阶段不能由模型跳过或通过提示词改写。

每次调用先写入不可变 reservation，再记录严格响应、状态变化和内容寻址的制品引用。事件包含连续
序号、前序哈希和完整事件哈希，`replay_agent_run()` 只折叠事件、不重新调用工具，因此不会重复模拟
成交或外部副作用。总调用次数、非法调用次数、核对重试次数、总时限和审批时限都冻结在 genesis
事件中；执行前取消或超时进入明确终态，提交结果不确定或核对发现严重差异则进入 `INCIDENT`。

当前 `InMemoryAgentRunRepository` 与精确响应重放只适合测试和单进程装配。生产级跨进程 CAS 仓储、
原始载荷留存/脱敏、outbox 和完整审计聚合仍由 P6-T09 负责。状态、工具与持久化边界详见
[显式 Agent Runtime 说明](./packages/quant_agent/agent/runtime/README.md)。

## P1 数据底座

P1 已提供：

- SQLAlchemy 数据模型与 Alembic 初始迁移；
- 交易日历、稳定证券 ID、历史代码和状态区间；
- Provider 协议、Fake Provider 和 Tushare 日线 HTTP 适配器；
- 原始响应归档与幂等日线入库；
- 独立复权因子、公司行动和复权研究视图；
- Point-in-time 财务修订查询；
- 历史行业归属；
- 数据质量规则和失败关闭；
- Parquet 不可变快照与 DuckDB 查询。

创建或升级本地数据库：

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
```

Alembic 会自动创建文件型 SQLite 的父目录，并优先读取
`QUANT_AGENT_DATABASE_URL`。如需可选的本地 PostgreSQL：

```powershell
docker compose up -d postgres
$env:QUANT_AGENT_DATABASE_URL = 'postgresql+psycopg://quant_agent:change-me-local-only@localhost:5432/quant_agent'
.\.venv\Scripts\python.exe -m alembic upgrade head
```

测试默认使用内存数据库、伪数据和 HTTP Mock，不会访问真实行情接口。真实调用
Tushare 时，Token 必须从密钥引用解析，不得写入仓库或日志。

### Tushare RAW-first 增量同步

先升级数据库，再通过进程环境提供 Token：

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
$env:MARKET_DATA_TOKEN = '<your-token>'
```

建议按“日历 → 主数据 → 日线 → 复权因子”的依赖顺序同步。例如：

```powershell
.\.venv\Scripts\quant-agent.exe data sync --dataset calendar --market SSE `
  --start 2026-08-28 --end 2026-08-28
.\.venv\Scripts\quant-agent.exe data sync --dataset instrument --instrument-type STOCK `
  --start 2026-08-28 --end 2026-08-28
.\.venv\Scripts\quant-agent.exe data sync --dataset daily --instrument-type STOCK `
  --market SSE --start 2026-08-28 --end 2026-08-28
.\.venv\Scripts\quant-agent.exe data sync --dataset adjustment --market SSE `
  --start 2026-08-28 --end 2026-08-28
```

`data sync` 当前支持交易日历、股票/ETF/指数主数据、未复权日线和股票复权因子。
每个成功响应先归档原始 JSON，再解码和写入 CURATED 表；失败运行可复用已归档原始页，
且每条整理记录可反查原始载荷。命令不接受明文 Token 参数。若不使用默认
`MARKET_DATA_TOKEN`，可通过 `--token-env <变量名>` 指定另一个环境变量。

查看运行、失败和检查点状态：

```powershell
.\.venv\Scripts\quant-agent.exe data status --provider tushare --limit 20
```

仓库中的验证只使用 HTTP Mock；由于当前开发环境未提供真实 Token，尚未声称完成
Tushare 在线验收。不同账户积分和接口权限也可能影响真实调用结果。

## 目录

```text
apps/                 API、任务、执行网关和 Web 入口
packages/quant_agent/ Python 领域与基础模块
configs/              环境、策略、风控和标的池配置
data_contracts/       跨服务数据契约
migrations/           数据库迁移
tests/                单元、集成、回放、对抗和端到端测试
docs/                 设计、开发与进度文档
```

## 安全提示

- 不要提交 `.env`、Token、券商凭证或真实账户数据；
- 本地默认运行模式为 `RESEARCH`；
- 配置加载器拒绝启用 `LIVE_AUTO`；
- `apps/execution_gateway` 将作为独立执行边界，不能被研究代码直接调用。
