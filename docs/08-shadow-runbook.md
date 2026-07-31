# Quant Agent P8-T07 影子运行手册

## 1. 当前状态

- 控制面状态：运行准备完成，等待首个交易日；
- 真实证据：`0 / 20` 个交易日；
- 计划窗口：2026-08-03 至 2026-08-28；
- 运行条件：凭证已通过本机受限文件注入；真实 Tushare 日历已冻结；
- 自动执行：2026-08-03 至 2026-08-28 的工作日 18:00 执行，非冻结交易日不记账。

交易日历已通过 Tushare `trade_cal` 冻结为 2026-08-03 至 2026-08-28 的 20 个交易日，
与计划窗口一致；日历 SHA-256 为
`aca4e7784a377a713fc3789886b190d22653ebad1e0ccd70a49db742fb59ea9f`。冻结结果保存在
本机忽略目录中，不覆盖既有证据。

## 2. 安全边界

- 运行模式只允许研究和影子计算，不产生可执行订单；
- `executable_orders_emitted` 必须始终为零，否则立即触发 P0；
- 每日输入必须具有数据快照 SHA-256 和虚拟账户快照 SHA-256；
- 数据可用时间不得晚于观察时间，未来数据计数必须为零；
- 虚拟账户每日执行订单、成交、现金和持仓核对；
- 账实不一致必须同步建立 P0/P1 事故，事故关闭前门禁保持阻塞；
- Agent 不得关闭事故、修改证据、恢复 Kill Switch 或补写历史日期；
- 市场数据令牌只从环境变量或只读密钥文件注入，不写入配置、日志、证据或报告；
- 容器运行使用 `MARKET_DATA_TOKEN_FILE`，避免令牌出现在容器环境检查结果中。

## 3. 首次启动

使用 Miniconda 环境：

```powershell
$python = 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe'
$env:MARKET_DATA_TOKEN = '<由密钥服务或当前终端安全注入>'

& $python .\scripts\shadow\bootstrap_calendar.py `
  --start 2026-08-03 `
  --required-days 20 `
  --market SSE `
  --output .\data\shadow\shadow-20260803-v1\calendar.json
```

只有真实提供商日历成功写入后，才允许累计第一个交易日。

容器采集时，将令牌单独保存到工作区外的本地只读文件，并设置
`MARKET_DATA_TOKEN_FILE`；`deploy/compose.shadow.collect.yml` 会把它挂载为
Docker Secret。状态重算不需要数据令牌，使用 `deploy/compose.shadow.yml`。

## 4. 每日运行顺序

收盘数据达到提供商可用时间后执行：

1. 获取交易日历并确认当日开市；
2. 获取全市场日线，保存原始快照和记录数；
3. 运行数据质量、市场阶段、主线、龙头和候选计算；
4. 在虚拟账户执行只读信号评估，不创建执行订单；
5. 完成订单、成交、现金和持仓核对；
6. 生成日报并输出严格的 `ShadowDayEvidence` JSON；
7. 追加证据账本、重算门禁、生成状态报告并投递异常告警。

真实数据采集命令：

```powershell
& $python .\scripts\shadow\collect_tushare_day.py `
  --trading-date 2026-08-03 `
  --market SSE `
  --output-dir .\data\shadow\shadow-20260803-v1\snapshots
```

追加完整每日证据：

```powershell
& $python .\scripts\shadow\record_day.py `
  --config .\configs\shadow\shadow_20260803_v1.toml `
  --calendar .\data\shadow\shadow-20260803-v1\calendar.json `
  --evidence .\data\shadow\shadow-20260803-v1\days\2026-08-03.json `
  --ledger .\data\shadow\shadow-20260803-v1\evidence.jsonl `
  --report-dir .\artifacts\shadow\shadow-20260803-v1 `
  --alert-log .\logs\shadow\shadow-20260803-v1-alerts.jsonl
```

每日证据必须包括：日期、观察时间、数据可用时间、输入快照、虚拟账户快照、数据版本、
市场阶段、主线、候选、数据/日报/任务结果、核对结果、未来数据计数、可执行订单计数、
事故、恢复记录和人工干预时长。

## 5. 状态与恢复

状态命令会从第一条记录开始校验序号、前序哈希和当前记录哈希，任何修改都会失败：

```powershell
& $python .\scripts\shadow\status.py `
  --config .\configs\shadow\shadow_20260803_v1.toml `
  --calendar .\data\shadow\shadow-20260803-v1\calendar.json `
  --ledger .\data\shadow\shadow-20260803-v1\evidence.jsonl `
  --report-dir .\artifacts\shadow\shadow-20260803-v1
```

- 同一日期、同一内容重试为幂等，不产生第二条记录；
- 同一日期不同内容会被拒绝，禁止覆盖；
- 后续日期已经写入后，不允许补写更早日期；
- 交易日缺口将状态改为 `BLOCKED`，后续天数不能掩盖连续性缺口；
- 失败日保留原记录，通过后续事故关闭记录反映修复，不删除历史；
- 证据损坏时保留原文件，复制后调查，不直接编辑生产账本。

## P0 executable order

立即停止影子会话并开启 Kill Switch；核查订单网关、工具权限与模式配置。确认不存在真实
委托后，由 RiskAdmin 和运行负责人共同关闭事故。该次会话是否作废需形成书面结论。

## P0 future data

隔离受影响快照和所有派生结果，定位数据可用时间、时区和查询边界。受影响日期不得计入
通过证据，不允许通过调整时间戳消除告警。

## P1 pipeline failure

保留失败证据、错误摘要和输入哈希。恢复任务必须使用相同输入版本；若跨过下一个交易日
仍未形成证据，连续运行门禁进入 `BLOCKED`。

## P1 missing day

停止累计后续通过天数，检查调度、主机、数据提供商和报告任务。缺口不能通过回放冒充
实时运行；需要重新开始一段完整的连续 20 个交易日窗口。

## P2 incomplete day

检查数据完成度和日报生成。允许保留少量失败日，但最终数据、日报及任务成功率必须达到
会话阈值。

## Incident handling

每个事故必须有唯一 `inc-` 编号、P0～P3 等级、组件和不含敏感信息的摘要。每日证据只
能引用事故开启或关闭；P0/P1 全部关闭前 P8-T07 不得完成。

## 6. 完成条件

只有同时满足以下条件，才将 P8-T07 改为 `已完成`：

- 连续交易日达到 20/20，且没有缺口；
- 未来数据违规为零；
- 可执行订单为零；
- 数据、日报和任务成功率达到配置阈值；
- 虚拟账户差异均有事故记录并已关闭；
- 主线和候选稳定性指标已生成；
- 所有 P0/P1 已关闭并留有修复证据；
- 最终 JSON/Markdown 报告和哈希链验证通过。
