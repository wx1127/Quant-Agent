# Production readiness checklist

这份清单用于区分“代码通过测试”和“系统获准进入真实环境”。任何一项未完成，都必须保持 `ExecutionEnvironment.LIVE` 门禁关闭。

## 必备证据

- [ ] GitHub Actions 的 `main` 分支 CI 成功记录；
- [ ] 至少 20 个交易日的 Shadow 运行日志，包含数据版本、决策 ID 和被抑制的副作用动作；
- [ ] 至少 3 个月 Paper 运行记录，包含信号、订单、模拟成交和账实核对；
- [ ] 券商沙盒连接测试和接口版本记录；
- [ ] 账户权限、程序化交易授权和合规报告确认；
- [ ] 风控阈值与券商限制的逐项映射；
- [ ] 监控、告警、日志脱敏和审计链演练记录；
- [ ] Kill Switch 触发、恢复和回滚演练记录；
- [ ] 小资金验收计划、人工审批人和紧急联系人；
- [ ] 每日亏损、单笔仓位和总风险上限已配置并验证。

## 进入 LIVE_ASSISTED 的硬门槛

必须同时满足：

1. `ComplianceChecklist.live_ready()` 返回 `True`；
2. `SmallCapitalAcceptance.validate(...)` 通过；
3. 每笔订单均绑定决策 ID、证据和人工审批令牌；
4. Paper 最近周期无未解决核对差异；
5. Kill Switch 和回滚流程在当前发布版本上演练成功。

## 禁止事项

- 不得用本地单元测试替代券商沙盒验证；
- 不得在缺少授权或合规确认时打开 LIVE 开关；
- 不得把 Shadow/Paper 的成功结果描述为真实收益；
- 不得删除失败运行、拒绝审批或 Kill Switch 事件；
- 不得通过修改前端绕过服务端审批和风控门禁。
