# 决策快照接入说明

本模块为一次 Harness 决策建立只写一次的身份边界。`DecisionSnapshot` 同时绑定决策模式、
市场、UTC 决策时点、已校验数据快照、策略配置、注册参数、组合风控、账户快照、代码制品、
Agent 和模型版本，并以完整内容哈希检测字段替换。它只证明“这次决策使用了哪些输入”，
不是下单许可，也不会授予 Agent 数据库或执行权限。

## 创建与恢复

应用应通过 `DecisionSnapshotService` 创建快照，不要直接把调用方提供的版本字符串拼成
`DecisionSnapshot`。服务会执行以下绑定：

- 从可信 `SnapshotStore` 获取“本次实际完成文件和内容哈希校验”的同一份清单，避免按版本
  重新加载时把另一个清单误绑定到决策；
- 对 `AccountSnapshot` 做严格 JSON 往返，并要求模式、`as_of` 和 `data_version` 与决策一致；
- 将真实 ETF 轮动或主线龙头配置与 `RegisteredParameterSet` 逐项核对，拒绝名称、版本、字段、
  值或 scalar 类型不一致；
- 从不可变 `PortfolioRiskPolicy` 读取策略版本与完整阈值哈希；
- 校验完整小写 Git SHA-1/SHA-256 和代码制品 SHA-256。

`resume()` 会重新解析当前所有来源；任一内容、哈希或版本变化都要求创建新的
`decision_id`。它不会把新数据静默装入旧决策。数据集的内容完整性不等于逐行 Point-in-time
正确性，后续确定性工具仍必须按冻结的 `decision.as_of` 过滤每条记录的可用时间。

## 存储与信任边界

`InMemoryDecisionSnapshotRepository` 用于测试和单进程临时运行；
`SQLiteDecisionSnapshotRepository` 用于本地持久化和重启恢复。相同 `decision_id` 只允许完整
规范 JSON 的精确幂等重放，任何字段差异都会冲突。SQLite 仓储使用写事务串行化并发创建，
以受控触发器拒绝 `UPDATE`、`DELETE` 及同主键 `INSERT`/`REPLACE`，底表使用
`WITHOUT ROWID` 避免通过隐藏行键替换另一身份；每次操作前都会复核 schema version、表和
完整触发器集合。读取时重新验证 JSON、内容哈希及关系行绑定，公共接口也不提供更新或删除。

仓储、SQLite 文件、数据快照目录以及生成代码制品哈希的构建流程均属于可信应用内部。
不得向 Agent 或不可信脚本提供直接写权限。SQLite 不是不可篡改账本：拥有数据库文件写权限的
主体仍可能替换一整行自洽内容，因此生产部署必须用操作系统权限、只写服务账号、备份或外部
不可变审计层保护该文件。

## 版本责任与当前边界

传入服务的 `AccountSnapshot`、`RegisteredParameterSet`、策略配置和风险政策必须来自可信应用
装配层。当前服务会验证这些对象内部的哈希和交叉关系，但尚不会查询独立的账户/参数制品注册表
来证明调用方提交的对象确实是注册表成员。

`code_artifact_hash` 应由部署流程对实际可执行制品计算，不能使用描述性占位值。
`agent_version` 应解析到固定的提示词、工具清单和 Harness 配置；`model_version` 应使用提供方的
不可变模型发布标识，不能使用 `latest`、`default` 等可漂移别名。核心契约会保存并哈希这些值，
但当前不会访问外部模型注册中心替调用方证明其真实性。

快照仓储只保存版本、ID 和内容哈希，不复制市场数据、账户明细、策略配置或模型制品。
对应制品必须由各自的可信存储按这些引用保留；从 `decision_id` 聚合所有制品和工具轨迹、执行
完整差异回放属于 P6-T09。P6-T02 及后续工具权限/状态机也仍是独立门禁。
