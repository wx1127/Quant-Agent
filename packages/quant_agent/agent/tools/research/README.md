# Agent 研究分析工具

本目录实现 P6-T03 的七个只读工具：

- `get_market_snapshot`
- `validate_market_data`
- `detect_market_regime`
- `rank_market_themes`
- `rank_theme_leaders`
- `rank_stock_candidates`
- `explain_candidate`

它们是已有确定性领域引擎的严格适配层，不是第二套评分实现。模型只能提交展示上限、是否展示
排除项和一个已有候选代码；`decision_id`、`as_of`、`data_version`、账户、配置及哈希全部来自
`ToolExecutionContext` 和可信装配。

## 计算链路

每次调用从冻结决策恢复同一批权威输入，并按阶段重新计算：

```text
verified SnapshotManifest
  -> QualityEngine(session=None, observed_at=decision.as_of)
  -> MarketRegimeClassifier（完整历史逐日分类）
  -> apply_regime_transitions（完整历史状态迁移）
  -> apply_mainline_states（完整历史主线持久性）
  -> LeaderEngine（账户规模绑定的可交易性）
  -> CandidateEngine（明确标记为未校准）
  -> explain_candidate（同一 CandidateSnapshot 的纯投影）
```

只计算最后一个交易日会丢失市场状态确认和主线持久性，因此 `ResearchPipeline` 要求市场特征与
行业特征都是严格递增、逐日对齐、无未来记录且最后一条精确等于决策时点。每个数据版本、分类
版本、配置哈希和上游结果哈希仍由领域契约继续校验。

## 可信输入边界

`ResearchInputSource` 是装配层必须实现的只读协议。它提供已验证清单、质量上下文、完整市场与
行业历史、账户绑定的龙头输入，以及可选的 PIT 候选补充信号。仓库目前没有统一的 Parquet 表
模式和 Arrow 到全部领域快照的生产解码器，因此本模块不会扫描数据文件、接受模型 SQL，或在
依赖缺失时注册占位 handler。生产装配应先实现并审查具体 source；依赖不完整时七个工具应保持
未注册。

输入对象会先经 JSON 往返重新验证，再检查：

- 清单的 `data_version` 与 `content_hash` 精确匹配 `DecisionSnapshot`；
- 质量上下文绑定同一数据内容，质量引擎不接收数据库会话；
- 市场和行业历史不越过 `as_of`，并以该时点结束；
- 龙头输入的账户 ID、账户快照 ID 与哈希完全匹配决策；
- 股票强度、可交易性和基本面覆盖相同证券集合，且基本面 `data_version` 不能漂移；
- 候选补充信号不能来自未来或另一数据版本。

可交易性容量使用账户规模，因此 `rank_theme_leaders`、`rank_stock_candidates` 和
`explain_candidate` 虽然仍是 `READ_ONLY / RESEARCH_ANALYSIS`，中央策略额外将其标记为
`account_scoped=True`。注册器必须获授决策中的账户 ID 才会暴露或执行这三项工具。

## 输出与失败语义

所有成功结果都是封闭、冻结的 `ToolOutput`，并由注册器再次序列化验证。输出重复携带决策、
数据、账户、模型、配置、输入和结果身份；数值不由 Agent 重算。市场置信度明确标记
`confidence_is_probability=false`，候选结果明确标记 `calibrated=false`，不得解释为预测概率或
交易建议。

正常候选、已评分但被硬门槛排除的候选、上游无法评分的 `CandidateExclusion` 是三种不同状态。
`explain_candidate` 只投影同一候选快照中的一条记录；未知代码返回 `NOT_FOUND`，绝不生成解释。

预期的可信输入缺失返回 `DATA_UNAVAILABLE`；版本、时间、哈希、账户或领域对齐失败返回
`DATA_INVALID`。失败响应始终包含 `errors`，不会用 `ok=true` 的空数组冒充“无信号”。未预期的
handler 异常仍由 P6-T02 注册器统一收敛为安全的 `ToolExecutionFailed`，不会泄露底层异常文本。

## 显式装配

`build_research_tools(source, ...)` 只构造七个 `AgentTool`，不会修改全局状态，也不会自动注册。
可信 composition root 应固定五类引擎配置，构造全部工具后逐一交给
`AgentToolRegistryBuilder.register()`。如果 source 或任一配置无效，构造应整体失败，避免只注册
一条不完整的研究链路。
