# Quant Agent

Quant Agent 是一个面向中国内地股票与场内 ETF 的日频量化研究和受控交易辅助系统。

系统采用以下边界：

- 大模型负责理解、编排与解释；
- 确定性程序负责行情、因子、回测和组合计算；
- 独立风控拥有订单否决权；
- 第一版实盘订单必须人工审批；
- 第一版明确禁用自动实盘。

## 当前进度

P0 本地工程基础、P1～P7 以及 P8 本地验证基础已实现；P0 的远程 CI 仍待在
GitHub 页面确认，P8 部署和长期运行任务尚未提前启动。
详细状态见：

- [项目进度文档](./docs/04-project-progress.md)

设计与开发资料：

- [Agent Harness 设计文档](./docs/01-agent-harness.md)
- [项目说明文档](./docs/02-project-overview.md)
- [项目开发文档](./docs/03-development-guide.md)

## Miniconda 环境

项目环境路径：

```text
D:\DevelopTool\MinConda\envs\Quant Agent
```

PowerShell 中激活：

```powershell
& 'D:\DevelopTool\MinConda\Scripts\activate'
conda activate 'D:\DevelopTool\MinConda\envs\Quant Agent'
```

也可以不激活，直接使用环境内的 Python：

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' --version
```

## 安装

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m pip install -r requirements-dev.lock
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m pip install -e .
```

## 质量检查

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m ruff format --check .
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m ruff check .
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m mypy packages apps
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' scripts/verify_ci_gates.py
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m pytest
```

CI 会在所有分支推送、面向 `main` 的 Pull Request 和手工触发时运行。手工触发可
选择 `format`、`type` 或 `test` 负向故障，验证对应质量门确实阻断；普通运行还会
在临时目录自动验证三类故障，并上传 JUnit 与覆盖率报告。

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
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m alembic upgrade head
```

测试默认使用内存数据库、伪数据和 HTTP Mock，不会访问真实行情接口。真实调用
Tushare 时，Token 必须从密钥引用解析，不得写入仓库或日志。

## P2 市场阶段与主线

P2 已提供：

- Point-in-time 安全的滚动特征、版本元数据和稳定缓存键；
- 主要指数 20/60/120 日趋势、均线位置和斜率；
- 市场上涨宽度、新高新低、成交额分位和下行风险；
- 上涨、强势震荡、分化、下跌、底部修复五类市场阶段；
- 状态确认、防抖、紧急下行切换和风险预算上限；
- 使用历史行业归属的 5/20/60 日行业相对强度与扩散；
- 主线形成、确认、拥挤和退潮状态机；
- 绑定冻结快照的时序回放、状态持续期和 Precision@K 评测。

## P3 龙头、候选与事件证据

P3 已提供：

- 可信来源公告的 Point-in-time 过滤、实体映射、内容哈希和去重；
- 将新闻、公告中的指令性文本作为不可信数据隔离并保留审计记录；
- 个股趋势、相对行业/基准强度、突破保持、回调和新高距离；
- 停牌、涨跌停、上市天数、成交活跃度和资金容量过滤；
- 当时已公告财务数据的质量评分、缺失状态和风险惩罚；
- 仅在已确认主线内运行的趋势/容量等龙头分类和评分；
- 候选 A/B/观察分层、排除原因、证据、观察条件和失效条件；
- 按年份、市场阶段、层级拆分的成本前后收益和 Precision@K 回放。

候选输出未经过概率校准，因此只展示分数和排名，不展示上涨概率。

## P4 策略与回测

P4 已提供：

- 与具体策略解耦且可序列化重放的信号、订单、成交和费用契约；
- 至少滞后一交易时点的向量化因子分层与基准比较；
- 维护现金、持仓、订单状态和股票 T+1 的事件驱动回测；
- 按生效日期版本化的交易单位、停牌、涨跌停、手续费、卖出税和滑点；
- ETF 动量、波动调整、趋势过滤、历史标的池和现金降仓策略；
- 已确认主线内趋势/容量龙头的目标组合、退出和换手约束；
- 训练、验证、冻结样本外与滚动前推切分，以及不可变参数注册；
- 同时展示成本前后收益、回撤、风险、换手、市场阶段和失败时期的报告。

策略模块只输出目标组合，不会直接创建或提交真实订单。

## P5 组合、风控与模拟交易

P5 已提供：

- 带可用/冻结资金和数量、估值时点及稳定哈希的不可变账户快照；
- ETF 与股票策略资金先隔离、再聚合的目标组合和差异计算；
- 独立于 Agent 和策略、服务异常时默认拒绝的结构化风控；
- 总仓位、个股、ETF、行业、换手、市场预算和回撤限制；
- 只在风控通过后产生且没有执行权限的确定性订单草案；
- 不接触真实券商凭证、支持幂等批次和 T+1 的模拟账户与撮合；
- 订单、成交、现金和持仓差异核对；
- 严重账实差异自动触发、只能由 RiskAdmin 复核恢复的 Kill Switch。

本阶段仍不提供真实券商下单和自动实盘能力。

## P6 Agent Harness

P6 已提供：

- 锁定数据、策略、参数、风控、账户和代码版本的不可变决策快照；
- 未注册即拒绝的工具注册器，以及运行模式、状态和参数权限校验；
- 市场、主线、龙头、候选、组合、风控、订单草案和模拟盘的受控工具适配；
- 不允许跳过风控、人工审批和成交核对的显式状态机；
- 最大工具调用次数、总超时、取消和完整状态轨迹；
- 事实、推断、反对证据、风险和失效条件分离的回答契约；
- 数字到工具响应字段的证据引用，以及证据不足时的明确拒答；
- 提示注入隔离、对话切换模式拒绝、标的白名单和敏感信息过滤；
- 十二个固定黄金场景的可重复轨迹评测；
- 按决策 ID 重建原版本、工具记录和状态轨迹的审计回放。

Agent 仍不能审批订单、恢复 Kill Switch 或提交真实订单。P7 的独立人工入口不会
扩大 Agent 自身权限。

## P7 API、Web 与报告

P7 已提供：

- 使用 Bearer 认证、角色与账户范围权限、请求 ID 和统一错误结构的 `/v1` API；
- 市场阶段、主线、龙头、候选和个股正反证据查询；
- 只允许已发布策略与参数、不会阻塞请求线程的异步回测任务；
- 账户快照、服务端独立风控和明确不可执行的组合提案；
- 仅 Approver 可签发、绑定订单哈希且最多五分钟有效的一次性审批令牌；
- 绑定具体草案的幂等模拟提交、Kill Switch 门禁和核对入口；
- 数字来自结构化字段、同时展示正反证据和前日变化的 Markdown/HTML 日报；
- 市场研究、个股证据、回测、组合、风险、人工审批和受控对话 Web 页面。

启动本地服务：

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m uvicorn apps.api.app:app
```

如需在 Web 工作台展示已经生成的历史影子结果，可在启动前仅为本地模式注入分析文件和
一次性只读令牌：

```powershell
$env:QUANT_AGENT_APP_ENV = 'local'
$env:QUANT_AGENT_RUNTIME_MODE = 'RESEARCH'
$env:QUANT_AGENT_LOCAL_RESEARCH_PATH = '<历史影子目录>\days\2026-07-28.analysis.json'
$env:QUANT_AGENT_LOCAL_API_TOKEN = '<随机生成的本地令牌>'
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m uvicorn apps.api.app:app
```

使用 `http://127.0.0.1:8000/web/research#token=<本地令牌>` 首次进入；页面读取令牌后会
立即从地址栏移除片段并自动加载市场阶段、主线、龙头、候选和证据。该便捷入口只在
`local` 环境生效，不会为生产环境创建默认身份。

默认应用不会内置任何访问令牌或真实账户。部署时必须从外部认证与密钥服务注入用户；
Web 只提供模拟提交，真实券商执行仍未实现。

## P8 系统验证基础

P8 当前已完成无需生产资源的四项验证：

- 固定数据、PAPER 模式、Fake Broker、自动重置与失败诊断的端到端环境；
- 十二个执行真实组件、带逐项观测证据的系统黄金场景及机器可读回放报告；
- 审批角色矩阵、提示注入、敏感输出、Kill Switch 和明文凭证扫描；
- 特征排名延迟、内存保留、收盘后预算和一年数据容量验收。

常用命令：

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m tests.e2e.runner
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m tests.replay.runner
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m pytest tests/security
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m tests.performance.runner
```

详细边界与结果见 [P8 系统验证报告](./docs/05-p8-validation-report.md)。P8-T05
已完成：不可变镜像、前向迁移、Compose 部署、健康检查、失败回滚和非空审计保留
均已通过 Docker 实机验收。详见
[部署与回滚运行手册](./docs/06-deployment-runbook.md)。影子运行、
三个月模拟盘、券商合规与实盘辅助尚未完成。

P8-T06 监控与告警已完成：提供数据完成度、服务成功率/延迟、风控拒绝、订单、
账实核对和 Kill Switch 指标，重复下单与严重账实差异固定为 P0，风控不可用固定为
P1。Prometheus 规则、Grafana 看板及处置流程见
[监控与告警运行手册](./docs/07-monitoring-alerting-runbook.md)。

P8-T07 已使用2026年7月真实未复权日线完成20个交易日的严格时点历史影子运行，
状态为 `PASSED / 已完成`。45个行情快照和20条追加式哈希链证据通过复核，数据、日报和
流水线成功率均为100%，未来数据违规与可执行订单均为0。详见
[影子运行手册](./docs/08-shadow-runbook.md)和
[历史影子验收报告](./docs/09-p8-historical-shadow-report.md)。P8-T08 将于2026-08-03
开始不少于3个月的 PAPER 模拟盘验证。

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
