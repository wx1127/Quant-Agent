# P8-T08 模拟盘日终运行手册

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

- 项目级自动任务在工作日收盘后运行上述入口；
- 每次运行先核对前一交易日是否存在成功或失败证据；
- 缺少两者时必须追加“调度缺失”失败记录并告警；
- 自动任务不得执行 Git 合并、不得推送 `main`；
- 只有不少于3个月证据、账实一致、无重复订单、信号全量可追溯且高等级问题关闭后，
  P8-T08 才可标记为 `已完成`。
