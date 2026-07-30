# Configuration

- `environments/`：运行环境和安全默认值；
- `strategies/`：版本化策略参数；
- `risk/`：版本化风控规则；
- `agent/`：Agent 工具权限、调用预算和安全边界；
- `universes/`：版本化标的池。

禁止在策略代码中硬编码风控阈值，也禁止在环境配置中保存真实密钥。
