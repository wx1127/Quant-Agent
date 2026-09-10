# API

P7-T01 提供 `apps/api/core/app.py` 中的 FastAPI 应用工厂。运行开发服务：

```powershell
$env:PYTHONPATH="apps/api"
uvicorn core.app:app --reload
```

接口统一挂载在 `/v1`，使用 `Authorization: Bearer <subject>:<role>[,<role>...]` 进行本地开发认证；生产环境应替换为 OIDC/JWT 验证器。每个响应都会返回 `X-Request-ID`，错误响应使用 `{code, message, request_id}` 契约。

