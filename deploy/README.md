# Deployment assets

- `compose.test.yml`：使用命名数据卷和审计卷的隔离测试环境。
- `compose.production.yml`：只接受不可变镜像摘要和外部数据库连接。
- `../scripts/deploy/build-image.ps1`：构建镜像，可选推送并返回仓库摘要。
- `../scripts/deploy/deploy.ps1`：迁移预检、前向迁移、启动和健康检查。
- `../scripts/deploy/rollback.ps1`：只回滚代码镜像，不降级数据库、不替换审计卷。
- `../scripts/deploy/acceptance.ps1`：测试部署、进程故障、回滚和审计保留验收。

生产服务以非 root 用户、只读根文件系统、无 Linux capabilities 和
`no-new-privileges` 运行。生产启动会强制开启全局 Kill Switch。
