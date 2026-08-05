# P8-T08 模拟盘日终运行手册

## 当前候选与退出规则（E03）

- 候选评分前先过滤港股、涨跌幅限制高于 10% 的板块，以及当日收盘较前收盘涨跌幅绝对值超过 10% 的股票。过滤标的不参与评分、排序和 Top 10 截取。
- 被过滤股票只进入 `market.candidate_exclusions` 与 `next_day_order_draft.excluded_buy_candidates` 审计区，不得进入买入草稿。
- 退出层按收盘数据检查：成本回撤 8% 止损、盈利 20% 固定止盈、峰值盈利达到 10% 后从峰值回撤 8% 移动止盈、向下跌破 MA5/MA10、退出当前主线、低开 5%、跌幅 7% 且成交量达到前五日均量 1.5 倍、持仓满 20 个交易日，以及候选轮动退出。
- 强制风控优先于候选轮动；同一股票允许同时命中多条规则，证据必须保留全部原因及主规则。
- 所有退出信号在收盘后生成，只能在下一交易日模拟卖出。T+1 冻结仓位标记为 `BLOCKED_T_PLUS_ONE`，不得生成不可执行卖单或伪造成交。
- 新规则只作用于新生成的证据，不覆盖或改写已有成功日、失败日、账户及草稿文件。

## 0. 收盘收益报告要求

- 每个成功的 PAPER 交易日必须在 `reports/<trading_date>.json` 写入 `close_report`。
- `close_report` 必须使用当日收盘价盯市，包含期初权益、期末权益、当日收益、当日收益率、现金、持仓市值和逐只股票浮盈浮亏。
- 日报必须同时写入当日模拟成交 `filled_orders` 和下一交易日草稿 `next_day_order_draft`。
- 面向用户展示的成交、持仓、候选和草稿必须包含股票 `name`；`instrument_id` 仅作为审计和排错字段保留。
- 股票名称来自当天 `snapshots/instruments-<trading_date>.json`，缺失名称时允许回退到 `instrument_id`，但不得阻断 PAPER 验证。
- 次日买入草稿必须排除涨跌幅限制超过 10% 的板块：创业板 `CN.SZ.300*`、`CN.SZ.301*`、
  科创板 `CN.SH.688*`、`CN.SH.689*`、北交所 `CN.BJ.*`，并排除交易所段为 `HK` 的港股；被排除候选写入
  `next_day_order_draft.excluded_buy_candidates`。
- 买入排除规则不回写历史模拟成交。规则生效前已形成的 append-only 证据必须保留原状。

## 1. 运行边界

- 模式固定为 `PAPER`，执行器固定为无凭证、无券商端点的 `PaperBroker`；
- 开始日期为 2026-08-03，连续运行不少于3个月；
- 只在 Tushare 确认的交易日且当日 16:05 后运行；
- 当日收盘行情只生成下一交易日订单草稿；
- 下一交易日只使用上一交易日已经冻结的草稿，并按次日开盘价加配置滑点模拟成交；
- 不允许把失败日或漏跑日补写为成功日。

## 2. 每日证据

每个成功交易日在 `data/paper/p8-t08/` 下追加以下互相独立的证据：

- `snapshots/`：真实行情快照及哈希；
- `days/`：市场阶段、主线、候选、模拟订单和成交的完整日记录；
- `pending/`：仅允许下一交易日执行的订单草稿；
- `accounts/`：现金、持仓、成本和市值快照；
- `reports/`：费用、滑点、核对、告警及人工干预日报；
- `failures/`：调度、数据或流水线失败记录。

成功日文件和失败日文件都采用不可覆盖写入。同一日期已存在失败记录时，运行入口拒绝生成
成功记录。

人工明确要求当日恢复运行时，可以使用 `--retry-after-failure`。原失败文件必须保留，新增日记录
必须标记 `RECOVERED_AFTER_FAILURE`，且不能计为“干净调度成功”；恢复再次失败时写入独立的
`recovery_failures/`，不得覆盖首次失败。

## 3. 日终命令

```powershell
$python = 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe'
$env:PYTHONPATH = 'D:\User\文档\量化\packages;D:\User\文档\量化'
$env:MARKET_DATA_TOKEN_FILE = 'D:\DevelopTool\Quant-Agent\secrets\market_data_token.txt'

& $python .\scripts\paper\run_daily.py `
  --output-root .\data\paper\p8-t08 `
  --source-cache .\data\shadow\shadow-20260701-historical-v1\snapshots `
  --source-cache .\data\research\current\snapshots
```

退出码为0且同时存在当日日记录、账户快照、下一日草稿和日报时，才算一个有效模拟盘交易日。
非交易日会安全退出且不生成交易日证据。失败必须保留 `failures/<日期>.json`，不得删除后重跑
成成功日。

## 4. 调度与复核

- Windows 任务计划 `Quant-Agent-P8-Paper` 在每个工作日 16:00 启动
  `scripts/paper/run_scheduled.ps1`；
- 调度进程从16:00开始，等待至16:05点时数据安全门禁后调用日终入口；
- 任务设置为错过计划时间后尽快启动、禁止并发重复实例，并记录退出码及调度状态；
- 任务允许使用电池时启动且不会因切换到电池供电而被终止；安装或修复统一运行
  `scripts/paper/install_scheduler.ps1`，不得依赖手工参数；
- Codex 自动任务不再承担执行，只在16:45巡检当天成功或失败证据；
- 每次运行先核对前一交易日是否存在成功或失败证据；
- 缺少两者时必须追加“调度缺失”失败记录并告警；
- 自动任务不得执行 Git 合并、不得推送 `main`；
- 只有不少于3个月证据、账实一致、无重复订单、信号全量可追溯且高等级问题关闭后，
  P8-T08 才可标记为 `已完成`。
