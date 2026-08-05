# Quant-Agent 新会话交接说明（2026-08-05）

## 1. 交接目标

新会话继续执行 P8-T08 模拟盘验证，重点完成：

1. 处理 2026-08-05 新规则候选与订单的 append-only 重算预览；
2. 在 2026-08-06 PAPER 日终流程中执行用户指定的“早盘全部卖出创业板持仓”覆盖单；
3. 核对成交、费用、滑点、现金、持仓、收益、告警和账实一致性；
4. 更新项目进度并将功能提交推送到 feature 分支，绝不直接推送 `main`。

本任务始终是 PAPER 模拟盘。不得连接券商、不得真实下单、不得切换 LIVE。

## 2. 工作区与环境

- 仓库：`https://github.com/wx1127/Quant-Agent.git`
- 本地目录：`D:\User\文档\量化`
- 当前分支：`feature/p8-paper-validation`
- 远程跟踪：`origin/feature/p8-paper-validation`
- Python：`D:\DevelopTool\MinConda\envs\Quant Agent\python.exe`
- Python 路径：`D:\User\文档\量化\packages;D:\User\文档\量化`
- 行情凭证文件：`D:\DevelopTool\Quant-Agent\secrets\market_data_token.txt`
- 凭证只能通过 `MARKET_DATA_TOKEN_FILE` 或既有安全读取逻辑使用，严禁把令牌内容写入文档、日志、提交或对话。

新会话首先执行：

```powershell
git status --short --branch
git log -8 --oneline --decorate
```

截至最新更新，一次性重算脚本已经成功执行并删除；正常情况下工作区不应再出现 `_recalculate_20260805_once.py`。

## 3. 不可违反的安全与证据规则

- 执行器固定为 `PaperBroker`，模式固定为 `PAPER` 或只读 `PAPER_PREVIEW`。
- 当日收盘数据只能生成下一交易日草稿，不能产生同日成交。
- 明日开盘成交是模拟：日终取得完整日线后使用当日开盘价和既定滑点，成交时间记录为 09:32；不要声称系统在真实早盘连接券商成交。
- 所有信号只能使用当时已可用的数据，禁止未来数据泄露。
- `days/`、`accounts/`、`pending/`、`reports/`、`failures/` 和人工覆盖证据均为 append-only。
- 失败不得删除、覆盖或补写为成功；恢复运行必须保留原失败并标记 `RECOVERED_AFTER_FAILURE`。
- 原 `data/paper/p8-t08/pending/2026-08-05.json` 不得修改。
- 不得推送 `main`；每个功能阶段提交并推送 `feature/p8-paper-validation`。
- 用户界面和用户报告优先显示股票名称，代码只作为内部审计键。

## 4. P8-T08 当前证据状态

证据根目录：`data/paper/p8-t08/`

| 日期 | 状态 | 说明 |
|---|---|---|
| 2026-08-03 | 失败/调度缺失 | `failures/2026-08-03.json`；不得补写成功 |
| 2026-08-04 | 成功 | 首个完整 PAPER 成功日 |
| 2026-08-05 | 失败后恢复 | 原失败保留；日记录为 `RECOVERED_AFTER_FAILURE`，不计干净调度成功 |
| 2026-08-06 | 待执行 | 已生成用户授权的创业板全量退出覆盖单 |

2026-08-05 恢复日产生 5 笔买入模拟成交，账户持有：

| 股票 | 数量 | 8 月 5 日平均成本 | 8 月 5 日收盘价 |
|---|---:|---:|---:|
| 陇神戎发 | 11,300 | 13.110482 | 13.61 |
| 科蓝软件 | 14,200 | 10.458362 | 10.22 |
| 米奥会展 | 9,600 | 15.322250 | 16.54 |
| 中红医疗 | 9,500 | 15.432338 | 15.60 |
| 普联软件 | 8,300 | 17.463963 | 19.20 |

8 月 5 日收盘估值口径：

- 现金：264,691.11 元；
- 持仓市值：765,261.00 元；
- 总权益：1,029,952.11 元；
- 相对初始资金收益：29,952.11 元，约 +2.9952%。

已有历史证据不得因为后续规则修正而回写。

## 5. 2026-08-06 创业板全部退出覆盖单

用户在 2026-08-05 明确要求：“明天早盘将创业板的票全部卖出”。该指令只授权 PAPER 模拟盘。

覆盖单路径：

```text
data/paper/p8-t08/pending_overrides/2026-08-06.json
```

覆盖单内容：

| 股票 | 方向 | 数量 | 状态 |
|---|---|---:|---|
| 陇神戎发 | 卖出 | 11,300 | READY_AFTER_T_PLUS_ONE_RELEASE |
| 科蓝软件 | 卖出 | 14,200 | READY_AFTER_T_PLUS_ONE_RELEASE |
| 米奥会展 | 卖出 | 9,600 | READY_AFTER_T_PLUS_ONE_RELEASE |
| 中红医疗 | 卖出 | 9,500 | READY_AFTER_T_PLUS_ONE_RELEASE |
| 普联软件 | 卖出 | 8,300 | READY_AFTER_T_PLUS_ONE_RELEASE |
| 合计 | 卖出 | 52,900 | 5 笔 |

统一退出原因：`USER_DIRECTED_CHINEXT_EXIT`。

执行约束：

- 覆盖单 `mode=PAPER`、`manual_override=true`；
- 信号日为 2026-08-05，执行日为 2026-08-06；
- `PaperBroker` 在 8 月 6 日执行时先释放 T+1 冻结数量；
- 使用 8 月 6 日开盘价、5 bps 卖出滑点、0.03% 佣金（最低 5 元）和 0.05% 股票卖出税；
- 仍受停牌、零成交量、跌停、批次过期等市场规则约束；
- 原 8 月 5 日草稿为空批次，仍保留不变；其 SHA-256 为：
  `23fa69c4ff8c25a4c90d04f4de0f3d86d2c48aafd35ed0175d210c6b132cafb2`。

`scripts/paper/run_daily.py::_pending` 已支持覆盖单优先读取，并验证：

- 执行日一致；
- 显式 PAPER 模式；
- 显式人工授权；
- 完整列出同一执行日被替代的原信号日。

如果任一条件不满足，必须失败留痕，不能退回或拼接不确定批次。

## 6. 候选前置过滤规则

规则版本：`p8-paper-buy-exclusions-v2`。

过滤必须发生在评分、排序和 Top 10 截取之前。排除：

- 创业板：`CN.SZ.300*`、`CN.SZ.301*`；
- 科创板：`CN.SH.688*`、`CN.SH.689*`；
- 北交所：`CN.BJ.*`；
- 港股：交易所段为 `HK`；
- 当日收盘较前收盘涨跌幅绝对值超过 10% 的股票。

被排除股票不能拥有候选分数或候选排名，只能进入：

- `market.candidate_exclusions`；
- `next_day_order_draft.excluded_buy_candidates`。

如果过滤后没有合格股票，允许买入草稿为空，不得为了凑数量补入禁买股票。

## 7. 主线和退出规则

P8 主线必须使用当日冻结的 Tushare 行业字段，不得使用“沪市主板、深市主板、创业板、科创板、北交所”等交易所板块冒充主线。行业快照缺失时当日任务失败，不降级为交易所板块。

退出规则版本：`p8-paper-exit-rules-v1`。

| 规则 | 当前阈值 |
|---|---|
| 单股止损 | 收盘价相对平均成本收益 ≤ -8% |
| 固定止盈 | 收盘价相对平均成本收益 ≥ +20% |
| 移动止盈 | 峰值收益达到 +10%，随后从峰值回撤 ≥ 8% |
| MA5/MA10 退出 | 收盘价由上向下穿越 MA5 或 MA10 |
| 行业主线退出 | 持仓行业不再属于当前确认主线 |
| 大幅低开 | 开盘价较前收盘 ≤ -5% |
| 放量下跌 | 当日跌幅 ≤ -7%，且成交量 ≥ 前五日均量 1.5 倍 |
| 最大持仓时间 | 达到 20 个已评估交易日 |
| 候选轮动 | 持仓不在新的合规候选集合 |

强制风控优先于普通候选轮动。同一股票可以同时命中多条退出规则，证据必须保留全部原因和最高优先级主规则。

当天买入的 A 股不能同日卖出，但可以生成下一交易日卖出草稿；状态为 `READY_AFTER_T_PLUS_ONE_RELEASE`，次日执行前由 `PaperBroker` 解冻。

## 8. 已完成的一次性重算

用户曾要求“重新生成今天的候选股票模拟订单”。新规则重算需要完整 Tushare `stock_basic` 行业快照，首次请求因接口每小时一次的频率限制失败，且没有生成伪证据或部分成功文件。

为此创建的单次 heartbeat 自动任务已成功完成：

- Automation ID：`8-5-paper`
- 状态：单次执行完成后已删除，不再有后续运行
- 目标：限频恢复后运行 `scripts/paper/_recalculate_20260805_once.py`
- 输出目录：`data/paper/p8-t08/recalculations/`
- 预期文件：
  - `2026-08-05-rules-v2.json`
  - `instruments-2026-08-05-rules-v2.json`

重算使用 25 个冻结交易日行情和 5,537 只股票行业快照，只生成 `PAPER_PREVIEW`，没有覆盖 8 月 5 日原 `days/accounts/pending/reports`，也没有修改 8 月 6 日人工退出覆盖单。

重算结果：

- 行业主线前两名：铅锌、软件服务；
- 合规候选 10 只：泛微网络、北投科技、税友股份、盛达资源、用友网络、博彦科技、浪潮软件、达实智能、金徽股份、久其软件；
- 前置过滤项：204 只；
- 预览草稿：卖出原 5 只创业板持仓，并买入泛微网络 3,400 股；
- 预览退出原因：陇神戎发、米奥会展、中红医疗为行业主线退出，科蓝软件、普联软件为候选轮动退出；
- 实际调度仍使用 `pending_overrides/2026-08-06.json`，因此只执行 5 笔创业板卖出，不执行预览中的泛微网络买入。

原 8 月 5 日草稿哈希及人工退出覆盖单哈希均保持不变。一次性脚本已删除，`recalculations/` 中证据必须继续保留。

## 9. 调度状态

主执行器：Windows 计划任务 `Quant-Agent-P8-Paper`。

- 工作日 16:00 启动；
- 等待到 16:05 后执行日终流程；
- 允许电池供电；
- 错过计划时间后尽快启动；
- 禁止并发重复实例；
- 调用 `scripts/paper/run_scheduled.ps1`；
- 只使用 `PaperBroker`。

Codex 自动任务 `quant-agent-p8` 在工作日 16:45 只负责结果巡检，不承担主执行。

注意：`logs/p8-paper/scheduler-status.json` 交接时仍停留在 2026-08-04 的成功状态，因此不能作为 8 月 6 日成功依据。8 月 6 日必须检查新的 Windows 任务结果以及当天 `days/reports/failures` 文件。

## 10. 近期关键提交

| 提交 | 内容 |
|---|---|
| `81355e5` | 支持显式授权、append-only 的 PAPER 待执行覆盖单 |
| `120f8d0` | 修正 T+1：当日冻结仓位可生成次日卖出草稿，次日执行前解冻 |
| `1e42064` | PAPER 主线与退出改用真实行业，不再使用交易所板块 |
| `88553e6` | 候选前置过滤和完整退出规则层 |
| `c597d02` | 排除涨跌幅限制高于 10% 的板块 |
| `01867fe` | 排除创业板和港股 PAPER 买入 |
| `bb96f85` | 增加收盘盯市收益报告和股票名称 |

最新全量质量门禁：

- `pytest -q`：178 passed；
- 总覆盖率：90.98%；
- `ruff check`：通过；
- `mypy packages apps`：通过。

## 11. 新会话推荐执行顺序

### 立即执行

1. 阅读本文件、`docs/04-project-progress.md`、`docs/10-p8-paper-runbook.md` 和 `docs/11-p8-daily-profit-report.md`。
2. 执行 `git status --short --branch`，确认位于 `feature/p8-paper-validation`。
3. 检查第 8 节的重算证据仍存在，确认它只作为预览且没有进入实际调度。
4. 检查覆盖单存在且仍包含 5 笔、合计 52,900 股卖单。
5. 重新调用 `scripts.paper.run_daily._pending(...)` 验证 2026-08-06 选择的是覆盖单。

### 2026-08-06 16:05 后

1. 检查 Windows 计划任务是否完成；若调度未启动，先记录真实失败，不得补写成功。
2. 检查：
   - `days/2026-08-06.json`
   - `accounts/2026-08-06.json`
   - `pending/2026-08-06.json`
   - `reports/2026-08-06.json`
   - `failures/2026-08-06.json`
3. 核对覆盖单产生 5 笔 SELL 模拟成交；若某只因停牌、跌停或无行情被拒绝，真实记录拒绝，不得写成成功成交。
4. 核对成交时间、开盘价滑点、佣金、卖出税、现金变化和账实一致。
5. 核对创业板剩余持仓是否为零；未清零时说明具体股票、数量和原因。
6. 生成当日收益报告和下一交易日草稿，用户展示只用股票名称。
7. 将 `P8-T08-E04-04`、`E04-05` 按真实结果改为“已完成”或失败/待处理，不能提前完成。
8. 运行全量测试，提交并推送 `feature/p8-paper-validation`。

## 12. 常用命令

```powershell
$python = 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe'
$env:PYTHONPATH = 'D:\User\文档\量化\packages;D:\User\文档\量化'
$env:MARKET_DATA_TOKEN_FILE = 'D:\DevelopTool\Quant-Agent\secrets\market_data_token.txt'

& $python -m pytest -q
& $python -m ruff check packages tests scripts
& $python -m mypy packages apps
```

手工日终入口只应在真实交易日且 16:05 后使用：

```powershell
& $python .\scripts\paper\run_daily.py `
  --output-root .\data\paper\p8-t08 `
  --source-cache .\data\shadow\shadow-20260701-historical-v1\snapshots `
  --source-cache .\data\research\current\snapshots
```

不要使用 `--retry-after-failure`，除非用户明确要求当日失败后恢复运行；即使恢复，原失败文件也必须保留。

## 13. 完成标准

本交接并不表示 P8-T08 已完成。P8-T08 仍需连续运行不少于 3 个月，并满足：

- 每日证据完整；
- 账实一致；
- 无重复订单；
- 所有信号可追溯；
- 无未来数据泄露；
- 无 LIVE 连接；
- 高等级问题关闭。

只有全部门禁通过后，才能把 P8-T08 标记为“已完成”。
