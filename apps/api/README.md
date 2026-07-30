# API

FastAPI 入口为 `apps.api.app:app`。开发环境启动：

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m uvicorn apps.api.app:app
```

所有业务接口位于 `/v1`，使用 Bearer 认证、`X-Request-ID` 和统一错误结构。
默认应用不注册任何用户令牌；部署时必须从外部密钥系统构建 `AuthService`，不得把
原始令牌写入配置或仓库。
