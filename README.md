# Quant Agent

Quant Agent 是一个面向中国内地股票与场内 ETF 的日频量化研究和受控交易辅助系统。

系统采用以下边界：

- 大模型负责理解、编排与解释；
- 确定性程序负责行情、因子、回测和组合计算；
- 独立风控拥有订单否决权；
- 第一版实盘订单必须人工审批；
- 第一版明确禁用自动实盘。

## 当前进度

当前正在实施 P0 工程与治理基础。详细状态见：

- [项目进度文档](./docs/04-project-progress.md)

设计与开发资料：

- [Agent Harness 设计文档](./docs/01-agent-harness.md)
- [项目说明文档](./docs/02-project-overview.md)
- [项目开发文档](./docs/03-development-guide.md)

## Miniconda 环境

项目环境路径：

```text
D:\DevelopTool\MinConda\envs\Quant Agent
```

PowerShell 中激活：

```powershell
& 'D:\DevelopTool\MinConda\Scripts\activate'
conda activate 'D:\DevelopTool\MinConda\envs\Quant Agent'
```

也可以不激活，直接使用环境内的 Python：

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' --version
```

## 安装

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m pip install -r requirements-dev.lock
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m pip install -e .
```

## 质量检查

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m ruff format --check .
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m ruff check .
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m mypy packages
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m pytest
```

## 目录

```text
apps/                 API、任务、执行网关和 Web 入口
packages/quant_agent/ Python 领域与基础模块
configs/              环境、策略、风控和标的池配置
data_contracts/       跨服务数据契约
migrations/           数据库迁移
tests/                单元、集成、回放、对抗和端到端测试
docs/                 设计、开发与进度文档
```

## 安全提示

- 不要提交 `.env`、Token、券商凭证或真实账户数据；
- 本地默认运行模式为 `RESEARCH`；
- 配置加载器拒绝启用 `LIVE_AUTO`；
- `apps/execution_gateway` 将作为独立执行边界，不能被研究代码直接调用。
