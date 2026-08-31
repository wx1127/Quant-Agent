# Quant Agent

Quant Agent 是一个面向中国内地股票与场内 ETF 的日频量化研究和受控交易辅助系统。

系统采用以下边界：

- 大模型负责理解、编排与解释；
- 确定性程序负责行情、因子、回测和组合计算；
- 独立风控拥有订单否决权；
- 第一版实盘订单必须人工审批；
- 第一版明确禁用自动实盘。

## 当前进度

P0 工程基础、P1 数据底座和 P1.5 可运行化已经实现；项目已进入 P2 市场阶段/
主线研究与 P4 回测基础开发。当前具备可恢复的 RAW-first 数据同步、Point-in-time
特征、五类市场状态、行业强度分析和回测事件契约。策略、组合交易、Agent、API
和 Web 仍未完成，不能把当前版本视为可自动实盘系统。详细状态见：

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
