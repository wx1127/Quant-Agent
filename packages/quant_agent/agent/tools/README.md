# Agent 工具注册与权限边界

本模块是 Harness 的工具安全内核，不是业务工具自动发现器。只有可信应用装配层显式构造的
`AgentTool` 才能交给 `AgentToolRegistryBuilder`；名称还必须存在于固定 v1 中央白名单。
注册器不扫描 `execution`、`risk`、`portfolio` 等领域包，也不把公共 Python 方法、插件或第三方
callable 自动变成 Agent 权限。构建后映射、运行模式、能力和账户授权均不可修改。

## v1 模式矩阵

中央策略是权限上限；具体 handler 没有注册时，即使名称在矩阵中也不可调用。

| 工具 | RESEARCH | BACKTEST | PAPER | LIVE_ASSISTED | LIVE_AUTO |
|---|:---:|:---:|:---:|:---:|:---:|
| `get_market_snapshot` | ✓ | ✓ | ✓ | ✓ | — |
| `validate_market_data` | ✓ | ✓ | ✓ | ✓ | — |
| `detect_market_regime` | ✓ | ✓ | ✓ | ✓ | — |
| `rank_market_themes` | ✓ | ✓ | ✓ | ✓ | — |
| `rank_theme_leaders`¹ | ✓ | ✓ | ✓ | ✓ | — |
| `rank_stock_candidates`¹ | ✓ | ✓ | ✓ | ✓ | — |
| `explain_candidate`¹ | ✓ | ✓ | ✓ | ✓ | — |
| `run_backtest` | — | ✓ | — | — | — |
| `get_portfolio_snapshot` | — | — | ✓ | ✓ | — |
| `build_target_portfolio` | — | — | ✓ | ✓ | — |
| `check_portfolio_risk` | — | — | ✓ | ✓ | — |
| `create_order_draft` | — | — | ✓ | ✓ | — |
| `get_order_draft` | — | — | ✓ | ✓ | — |
| `submit_paper_orders` | — | — | ✓ | — | — |
| `submit_approved_orders` | — | — | — | ✓ | — |
| `reconcile_account` | — | — | ✓ | ✓ | — |
| `generate_decision_report` | ✓ | ✓ | ✓ | ✓ | — |

¹ 只读研究能力不变，但可交易性容量依赖冻结账户规模，因此还必须通过决策账户作用域校验。

`LIVE_AUTO` 在构建注册器时即被拒绝。`approve_order_batch`、切换模式、修改风控阈值和恢复或关闭
Kill Switch 都是人工/管理面操作，属于不可注册保留名。PAPER 与实盘提交使用不同名称，避免由
模型参数选择执行后端。

## 调用顺序与快照绑定

调用入口先验证精确工具名、是否实际注册、服务端模式和可信 capability；未注册或模式不允许的
请求不会触发昂贵的决策恢复，也不能借参数错误探测隐藏 schema。通过静态门禁后，注册器只调用
一次 `VerifiedDecisionResolver`。生产装配必须把它接到会重新核验当前数据、账户、策略、参数、
风险与代码制品的 `DecisionSnapshotService.resume()` 语义，普通仓储读取不够。

模型只提供工具名及工具 JSON 参数。`mode`、`request_id`、`decision_id`、`as_of`、
`data_version`、`account_id`、principal、capability 和审批均来自可信上下文。解析出的快照必须与
注册器模式和调用决策 ID 一致，且不能晚于请求边界；账户型工具还要匹配构建时授予的账户集合。
handler 得到的是重新序列化验证后的独立快照。

P6-T03 已实现七个研究 handler。它们只接受展示上限、排除项开关或一个已有候选代码；完整
历史、数据和账户身份均由可信 `ResearchInputSource` 提供。具体计算、失败语义及尚待接入的生产
快照解码边界见[研究分析工具说明](./research/README.md)。

## 参数与输出

所有参数模型继承 `ToolArguments`，输出数据模型继承 `ToolOutput`。二者都严格、冻结并拒绝额外
字段。注册阶段会冻结输入和输出 JSON Schema，并递归拒绝：

- 开放的嵌套模型、自由键 `dict`、`Any` 或无约束数组项；
- 非 lower-ASCII snake-case 字段或别名；
- 模型可控的模式、账户、决策、请求和时间字段；
- 审批、token、password、credential、secret 与 API key 字段。

原始 JSON 在 Pydantic 前先限制字节数、深度、节点数、对象字段数、数组长度和字符串长度，并
拒绝重复键、非有限数、循环、非 JSON 对象和超范围整数。随后使用严格 JSON 语义校验，因此
JSON 原生的枚举、日期时间和 Decimal 字符串仍可到达强类型模型，而字符串数字不会冒充整数、
整数也不会冒充布尔值。写操作必须有非空、有界、可打印且精确的 `idempotency_key`。

handler 必须返回标准 `ToolResponse`，其 request、decision、`as_of` 与数据版本必须精确绑定
授权快照；成功响应必须携带注册的精确 `ToolOutput` 类型。注册器重新序列化验证输出数据，并把
响应复制为基础标准响应，避免子类额外字段、旧模型实例或 handler 异常内容泄漏。

## 实盘授权与审计

实盘工具不能把审批令牌放进参数。可信 `LiveExecutionAuthorizer` 返回一次性 context-manager
授权作用域：进入时必须原子核验并预留/消费与本次决策和订单批次绑定的审批，授权在同步写入
`ALLOWED` 审计和 handler 执行期间持续有效，退出后释放。只返回一个布尔值不满足该契约，因为
校验与执行之间会产生撤销竞态。真实审批服务和券商执行网关由 P7/P8 接入；在它们完成前不要
注册 `submit_approved_orders` handler。

每次真实调用的允许或拒绝决定都会同步写入注入的 `AuditSink`。审计只保存固定枚举、版本、
快照哈希、参数哈希和安全校验码，不保存原始参数、验证器异常、审批令牌或未知工具原文。拒绝
审计失败时抛出 `ToolAuditUnavailable`；允许审计失败时 handler 同样不会运行。本地 JSONL sink
已保证单进程多线程整行追加，但它不是跨进程事务日志，也不提供 `fsync`、不可篡改存储或完整
调用回放；这些生产能力属于 P6-T09。

## 可信装配责任

注册器可以验证名称、schema、权限标签、上下文和结果，却无法从任意 Python callable 推断其
真实副作用。因而 `AgentToolRegistryBuilder`、resolver、audit sink、授权器和 handler 代码都在
可信计算基内：具体 wrapper 必须逐个代码审查，并证明其实际效果与中央 `name → capability →
effect` 策略一致。禁止接受模型生成代码、用户上传模块、通用插件方法或运行时反射结果作为
handler。P6-T03 的研究 wrapper 和后续 P6-T04 的组合/模拟交易 wrapper 都只能在 composition
root 显式注册，并继续依赖领域层的 PIT、风控、Kill Switch、账户模式和执行网关二次校验。
