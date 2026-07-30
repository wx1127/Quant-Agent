# Database migrations

数据库迁移由 Alembic 管理。

本地升级：

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m alembic upgrade head
```

生产环境必须通过显式连接配置运行迁移，不允许业务模块自行创建或修改表。

