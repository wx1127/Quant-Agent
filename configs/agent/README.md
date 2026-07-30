# Agent Harness 配置

`harness_v1.toml` 固定工具调用上限、任务超时、写操作幂等要求和运行模式权限。

- 运行模式只能由服务端环境配置决定，聊天内容不能切换模式；
- `approve_order_batch` 和 `recover_kill_switch` 只允许人工入口使用；
- `LIVE_ASSISTED` 不注册提交工具，`LIVE_AUTO` 保持禁用；
- 代码中的拒绝式权限校验是最终边界，配置不能扩大代码未允许的权限。
