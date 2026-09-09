# Risk configuration

`portfolio-v1.toml` 是首版组合风控阈值。所有小数必须使用带引号的精确十进制字符串，加载后生成完整策略哈希。

风险阈值不能由 Agent 修改；Agent 只能提交已绑定账户、组合提案、估值和策略哈希的风控请求。

Kill Switch 是带事故证据、修订号和审计链的运行时状态，不从 TOML 中读取一个布尔值来恢复交易。
GLOBAL 和 ACCOUNT 状态必须显式初始化；缺少任一状态或读取失败时拒绝新增订单。
恢复必须走 `KillSwitchService.recover()`，并注入可信的 `KillSwitchRecoveryAuthorizer`，
根据独立身份和审批记录验证整个请求。未注入、返回值不严格为 `True` 或认证服务异常均拒绝恢复。
仓储 API 属于可信应用装配层，不能注册为 Agent 工具，也不能把恢复入口作为 Agent 权限暴露。

