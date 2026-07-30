# Quant Agent 部署、回滚与发布清单

## 1. 安全边界

- 只部署 `repository@sha256:<64位摘要>`，禁止 `latest` 或可变标签；
- 基础镜像固定到 Python 3.12.13 Debian slim 的 OCI 摘要；
- 生产模式固定为 `LIVE_ASSISTED`，`LIVE_AUTO` 仍由配置模型拒绝；
- 生产进程启动时自动触发全局 Kill Switch，只有 RiskAdmin 复核后才能恢复；
- 生产必须配置独立的追加式审计路径；
- 迁移只允许向前升级，自动回滚永不执行数据库降级；
- 审计卷不随容器和代码镜像替换或删除。

## 2. 前置条件

- Docker Engine 与 Docker Compose；
- 可访问的镜像仓库；
- 测试环境数据库或 Compose 测试卷；
- 生产数据库连接通过安全环境注入；
- 已通过当前提交的 CI；
- 发布人、审批人和 RiskAdmin 已明确。

当前开发机已在 `D:\DevelopTool\Docker` 安装 Docker Desktop 4.84.0，使用 WSL 2
Linux Engine 29.6.2 和 Docker Compose 5.3.1；WSL 数据盘位于
`D:\DevelopTool\Docker\wsl-data`。P8-T05 已完成本地进程、迁移、容器构建、
Compose 部署、健康检查和失败回滚验收。

## 3. 构建不可变镜像

```powershell
.\scripts\deploy\build-image.ps1 `
  -Repository 'registry.example/quant-agent' `
  -Version '0.9.0' `
  -GitCommit '<完整提交哈希>' `
  -Push
```

脚本必须返回 `registry.example/quant-agent@sha256:...`。后续部署只使用该摘要。

## 4. 数据库预检

连接信息只从环境读取，不得写入命令行日志：

```powershell
$env:QUANT_AGENT_DATABASE_URL = '<由密钥服务注入>'
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' `
  .\scripts\deploy\migrate.py --check
```

返回码 `0` 表示已处于目标版本；返回码 `2` 表示需要前向升级；返回码 `1` 表示迁移
图不兼容，必须停止发布。升级由隔离的 `migrate` 服务执行。

## 5. 测试环境部署与验收

```powershell
.\scripts\deploy\deploy.ps1 `
  -ImageRef 'registry.example/quant-agent@sha256:<摘要>' `
  -Environment test
```

部署顺序为：校验 Compose、运行前向迁移、启动候选镜像、检查 `/health/ready`。

验证失败发布可以安全回滚：

```powershell
.\scripts\deploy\acceptance.ps1 `
  -CandidateImage 'registry.example/quant-agent@sha256:<候选摘要>' `
  -PreviousImage 'registry.example/quant-agent@sha256:<前一摘要>'
```

验收脚本会先追加专用审计标记，再停止候选进程模拟故障，启动前一不可变镜像，
并确认服务恢复、回滚前至少存在一条审计记录且回滚后记录数没有减少。

## 6. 生产发布

```powershell
$env:QUANT_AGENT_DATABASE_URL = '<由密钥服务注入>'
.\scripts\deploy\deploy.ps1 `
  -ImageRef 'registry.example/quant-agent@sha256:<已验收摘要>' `
  -Environment production `
  -PreviousImageRef 'registry.example/quant-agent@sha256:<前一摘要>'
```

发布后先保持全局 Kill Switch 开启。RiskAdmin 必须核对版本、迁移、数据、账户快照、
告警和审计记录，才能通过独立恢复流程解除门禁。

## 7. 回滚策略

回滚只切换到与当前数据库结构兼容的旧代码镜像：

```powershell
.\scripts\deploy\rollback.ps1 `
  -ImageRef 'registry.example/quant-agent@sha256:<前一摘要>' `
  -Environment production
```

如果旧镜像不兼容当前数据库版本，停止自动回滚，保持 Kill Switch，并按事故流程
处理。任何情况下都不得自动运行 `alembic downgrade`、删除数据库卷或清空审计卷。

## 8. 发布清单

- [ ] 功能分支 CI 成功；
- [ ] 镜像基础层与最终镜像均使用不可变摘要；
- [ ] 镜像提交哈希、配置版本和迁移版本已记录；
- [ ] 镜像扫描没有未关闭高危缺陷；
- [ ] 数据库迁移预检通过且有备份/恢复责任人；
- [ ] 测试环境部署和健康检查通过；
- [ ] 失败发布回滚演练通过；
- [ ] 回滚前后的历史审计记录完整；
- [ ] 生产 Kill Switch 启动状态为开启；
- [ ] 监控、告警和事故联系人已准备；
- [ ] 发布人与审批人完成双人复核；
- [ ] 发布结果及异常写入审计记录。
