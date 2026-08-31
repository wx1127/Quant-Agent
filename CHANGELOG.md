# Changelog

## Unreleased — 真实数据、市场状态与回测基础

- 新增带租约、检查点、幂等键、分页审计和失败重放的 RAW-first 增量同步；
- 新增 Tushare 股票/ETF/指数主数据与日线、交易日历和股票复权因子同步；
- 新增整理记录到原始响应的可追溯血缘及 0002～0004 数据库迁移；
- 新增 `quant-agent data sync`、`data status` 和 Point-in-time `research demo`；
- 新增滚动特征框架、市场趋势、市场宽度、行业强度和五类市场状态；
- 新增市场状态确认、防抖、严重下跌快速路径和可验证事件哈希链；
- 新增不可变回测事件、交易规则接口、版本化费用规则和确定性回放契约。

## 0.3.0 — P1.5 可运行化

- 新增 `quant-agent` / `python -m quant_agent` 命令入口；
- 新增确定性的 Provider 到 Parquet/DuckDB 演示流水线；
- 接通受控环境变量覆盖、`env://` 与可注入的 `vault://` 解析；
- 本地默认改为零配置 SQLite，并修复 Alembic 目录创建与 URL 覆盖；
- 增加 PostgreSQL Psycopg 驱动、Windows 时区依赖与可选 Compose 服务；
- 增强快照路径校验、数据库登记恢复和冲突检测。

## 0.2.0 — P1 数据底座

- 新增 SQLAlchemy 数据模型和 Alembic 初始迁移；
- 新增交易日历、稳定证券 ID、历史代码和证券状态区间；
- 新增统一 Provider 契约、Fake Provider 和 Tushare 日线适配器；
- 新增原始响应归档和幂等日线入库；
- 新增独立复权因子、公司行动及前/后复权研究视图；
- 新增 Point-in-time 财务修订查询；
- 新增历史行业分类与成员关系；
- 新增数据质量引擎；
- 新增 Parquet 不可变快照、内容校验和 DuckDB 查询。

## 0.1.0 — P0 工程基础

- 初始化项目结构和 Miniconda 开发环境；
- 建立依赖、格式、静态检查、类型检查、测试和 CI；
- 建立公共领域契约；
- 建立环境配置、密钥引用、日志、追踪、脱敏和审计基础。

