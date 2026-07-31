# P8-T07 历史时点影子运行验收报告

## 1. 验收结论

- 会话：`shadow-20260701-historical-v1`
- 模式：`HISTORICAL_POINT_IN_TIME`
- 目标窗口：2026-07-01 至 2026-07-28，共20个交易日
- 前置历史：25个交易日
- 真实行情快照：45个交易日
- 结果：`PASSED`
- 账本末尾哈希：`65239d97e36316d537a82f70857afb7ad5196f465d9b6288a46ed554c4c3b1e8`

P8-T07 按项目决策由实时影子运行改为严格时点历史影子运行。P8-T08 从
2026-08-03 起直接进入不少于3个月的 PAPER 模拟盘验证。

## 2. 防泄漏措施

- 每个交易日使用当日 `16:05 Asia/Shanghai` 作为独立虚拟时钟；
- 仅查询 `trade_date <= 当日` 且 `available_at <= 虚拟时钟` 的数据；
- 使用未复权日线，不使用未来复权因子；
- 每日从历史窗口重新计算市场阶段、主线和候选；
- 下载数据中晚于虚拟时钟的记录只统计为排除项，不进入特征、候选或证据哈希；
- 历史影子与实时影子模式不可混写同一会话；
- 证据按交易日顺序写入不可覆盖的 SHA-256 哈希链。

当前凭证没有 Tushare `index_member_all` 历史申万成员权限，因此没有使用当前行业归属
倒灌7月。主线限定为可由当日证券代码稳定识别的沪市主板、深市主板、科创板、创业板
和北交所板块，再在领先板块内按截至当日的5日/20日收益及成交额选择候选。Tushare
官方说明历史申万成员接口提供调入、调出日期，但需要单独权限：
<https://tushare.pro/document/2?doc_id=335>。

## 3. 汇总指标

| 指标 | 结果 |
|---|---:|
| 连续交易日 | 20/20 |
| 数据成功率 | 100% |
| 日报成功率 | 100% |
| 流水线成功率 | 100% |
| 每日证券数 | 5,511～5,526 |
| 主线平均稳定性 | 68.42% |
| 候选平均稳定性 | 33.13% |
| 未来数据违规 | 0 |
| 可执行订单 | 0 |
| 未关闭 P0/P1 | 0 |
| 凭证泄漏标记 | 0 |

## 4. 每日证据摘要

| 日期 | 证券数 | 市场阶段 | 领先板块 | 候选数 | 输入哈希前缀 |
|---|---:|---|---|---:|---|
| 2026-07-01 | 5511 | RANGE_STRONG | STAR, SH_MAIN | 10 | 188d08f497b7 |
| 2026-07-02 | 5517 | DOWNTREND | STAR, SH_MAIN | 10 | aa7929bd3ada |
| 2026-07-03 | 5516 | RANGE_STRONG | STAR, SZ_MAIN | 10 | 14534b90f3bb |
| 2026-07-06 | 5517 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | c99e08168afb |
| 2026-07-07 | 5517 | DOWNTREND | SH_MAIN, BJ | 10 | c071ea51eaf8 |
| 2026-07-08 | 5518 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | 293fb1a3394b |
| 2026-07-09 | 5520 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | 28752d9c197d |
| 2026-07-10 | 5521 | DIVERGENT | SH_MAIN, SZ_MAIN | 10 | 244bca7f72b1 |
| 2026-07-13 | 5524 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | af3181355ff8 |
| 2026-07-14 | 5524 | DIVERGENT | SH_MAIN, SZ_MAIN | 10 | b19d4aeb744e |
| 2026-07-15 | 5525 | DIVERGENT | SH_MAIN, SZ_MAIN | 10 | c1fb39a7971d |
| 2026-07-16 | 5524 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | dd365afce5a6 |
| 2026-07-17 | 5522 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | 6f3f26d7b037 |
| 2026-07-20 | 5524 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | c9ab320b2641 |
| 2026-07-21 | 5525 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | 54e7ebffec22 |
| 2026-07-22 | 5526 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | dd11dce4dbd8 |
| 2026-07-23 | 5526 | RANGE_STRONG | SH_MAIN, BJ | 10 | 5cc355ecb6fe |
| 2026-07-24 | 5526 | DOWNTREND | SH_MAIN, SZ_MAIN | 10 | 549bfcb4448c |
| 2026-07-27 | 5523 | RANGE_STRONG | CHINEXT, STAR | 10 | d0224d696875 |
| 2026-07-28 | 5524 | DOWNTREND | SZ_MAIN, SH_MAIN | 10 | 630e3b8ac8e8 |

## 5. 验收边界

本次结果验证了真实历史行情、严格时点过滤、连续逐日前推、市场阶段、板块主线、候选、
虚拟账户快照、哈希链、日报和安全门禁。它不验证真实盘后接口延迟、调度主机连续在线或
历史数据在原发布日期后的修订情况。这些运行问题转由 P8-T08 的3个月实时模拟盘覆盖。
