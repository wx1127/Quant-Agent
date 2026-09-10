# Deployment and rollback

部署以不可变 Git 提交和镜像标签为单位，禁止直接覆盖生产版本。

## 发布

```powershell
.\deploy\publish.ps1 -Version "2026.09.10-<git-sha>"
```

脚本会检查工作区干净、记录当前 SHA，并生成 `deploy/releases/<version>.json` 清单。真实环境可在此处接入镜像构建、迁移和编排平台。

## 回滚

```powershell
.\deploy\rollback.ps1 -Manifest deploy/releases/2026.09.10-<git-sha>.json
```

回滚前必须先通过健康检查；回滚后再次执行健康检查并保留操作日志。数据库迁移必须提供向后兼容路径，不能通过回滚应用代码来逆向破坏数据。
