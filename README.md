# Quant Agent

Quant Agent 是一个面向中国内地股票与场内 ETF 的日频量化研究和受控交易辅助系统。

系统采用以下边界：

- 大模型负责理解、编排与解释；
- 确定性程序负责行情、因子、回测和组合计算；
- 独立风控拥有订单否决权；
- 第一版实盘订单必须人工审批；
- 第一版明确禁用自动实盘。

## 当前进度

P0 工程基础、P1 数据底座和 P2 市场阶段与主线已经实现。详细状态见：

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
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m mypy packages
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m pytest
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
