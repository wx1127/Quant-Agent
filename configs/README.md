# Configuration

- `environments/`：运行环境和安全默认值；
- `strategies/`：版本化策略参数；
- `risk/`：版本化风控规则；
- `universes/`：版本化标的池。

禁止在策略代码中硬编码风控阈值，也禁止在环境配置中保存真实密钥。

`QuantAgentSettings.from_toml()` 用于纯验证；`QuantAgentSettings.load()` 额外应用受控的
`QUANT_AGENT_*` 环境变量。数据库 URL、行情 Token 和 LLM Key 应通过 `env://` 或
显式注入的 `vault://` resolver 在运行时解析。项目不会自动读取 `.env` 文件。

