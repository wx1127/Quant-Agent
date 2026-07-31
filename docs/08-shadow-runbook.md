# Quant Agent P8-T07 历史时点影子运行手册

## 1. 当前状态

- 会话：`shadow-20260701-historical-v1`
- 状态：`PASSED / 已完成`
- 证据：`20 / 20` 个连续交易日
- 窗口：2026-07-01 至 2026-07-28
- 模式：`HISTORICAL_POINT_IN_TIME`
- 后续阶段：P8-T08 于2026-08-03开始不少于3个月的 PAPER 模拟盘。

详细结果见[历史影子验收报告](./09-p8-historical-shadow-report.md)。

## 2. 验收口径

P8-T07 使用真实 Tushare 未复权日线进行严格时点逐日前推。每个交易日模拟当日
16:05 的信息边界，只允许使用当时已经可用的数据。历史影子验证算法和数据闭环；
真实延迟、调度稳定性及长期模拟成交由 P8-T08 覆盖。

验收要求：

- 连续不少于20个冻结交易日；
- 数据、日报和流水线成功率不低于98%；
- `available_at` 不得晚于当日虚拟时钟；
- 未来数据违规和可执行订单必须为0；
- 虚拟账户每日生成快照并完成核对；
- 不得存在未关闭的 P0/P1；
- 所有证据必须通过追加式哈希链校验。

## 3. 防未来数据规则

- `trade_date <= replay_date`；
- `available_at <= replay_date 16:05 Asia/Shanghai`；
- 不使用未来复权因子；
- 不使用今天的行业成员回填历史；
- 每日特征、主线和候选从截至当天的数据重新计算；
- 晚于虚拟时钟的数据只允许被排除，不允许进入输入哈希；
- 历史模式证据不得写入实时模式会话；
- 不允许跳日、覆盖、修改或逆序补写证据。

## 4. 主线与候选边界

当前凭证不具备历史申万行业成员接口权限。为保证没有成员关系倒灌，历史影子使用证券
代码可稳定确定的市场板块：

- `BOARD.SH_MAIN`：沪市主板；
- `BOARD.SZ_MAIN`：深市主板；
- `BOARD.STAR`：科创板；
- `BOARD.CHINEXT`：创业板；
- `BOARD.BJ`：北交所。

每天按板块成员截至当日的5日收益和成交额选择两个领先板块，再在其中按20日收益、
5日收益和成交额生成10只候选。此处“主线”是可审计的市场板块代理，不声称等同于完整
行业或概念题材主线。

## 5. 安全执行

行情凭证只能通过 `MARKET_DATA_TOKEN_FILE` 或临时环境变量注入，不得写入仓库、日志、
证据和报告。推荐路径位于仓库外：

```text
D:\DevelopTool\Quant-Agent\secrets\market_data_token.txt
```

禁止同时配置令牌值和令牌文件。历史影子不创建订单草稿、不调用真实交易接口，虚拟账户
保持只读影子状态，`executable_orders_emitted` 必须始终为0。

## 6. 重放命令

```powershell
$python = 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe'
$env:MARKET_DATA_TOKEN_FILE = 'D:\DevelopTool\Quant-Agent\secrets\market_data_token.txt'

& $python .\scripts\shadow\run_historical_replay.py `
  --config .\configs\shadow\shadow_20260701_historical_v1.toml `
  --start 2026-07-01 `
  --calendar .\configs\shadow\calendar_20260701_20d.json `
  --market SSE `
  --history-days 25 `
  --output-root .\data\shadow\shadow-20260701-historical-v1
```

重复运行会校验缓存快照哈希，并对完全相同的每日证据执行幂等返回。任何同日内容变化、
哈希损坏或日期逆序都会失败。

## 7. 证据文件

运行数据位于被 Git 忽略的本机目录：

```text
data/shadow/shadow-20260701-historical-v1/
├── calendar.json
├── snapshots/                 # 45个真实日行情快照
├── days/                      # 20份分析和20份证据
├── evidence.jsonl             # 20条哈希链账本
└── reports/
    ├── shadow-status.json
    └── shadow-status.md
```

仓库只提交配置、代码、测试和脱敏验收摘要，不提交体积较大的行情数据，也不提交凭证。

## 8. 故障处理

- Provider 限频：保留已写入且哈希通过的快照，稍后从缓存续跑；
- 快照哈希不一致：停止运行，保留原文件调查，不覆盖；
- 交易日无行情：停止目标会话，核对冻结日历；
- 未来数据进入查询：记为 P0，当前会话不得通过；
- 账实不一致：建立 P0/P1，关闭前保持阻塞；
- 凭证泄漏：立即吊销凭证、清理产物并进行历史扫描。
