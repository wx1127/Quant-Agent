# API

FastAPI 入口为 `apps.api.app:app`。开发环境启动：

```powershell
& 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe' -m uvicorn apps.api.app:app
```

所有业务接口位于 `/v1`，使用 Bearer 认证、`X-Request-ID` 和统一错误结构。
默认应用不注册任何用户令牌；部署时必须从外部密钥系统构建 `AuthService`，不得把
原始令牌写入配置或仓库。

研究接口包括：

- `/v1/market/regime`：市场阶段与正反证据；
- `/v1/themes`：真实行业或概念主线列表；
- `/v1/themes/{theme_id}`：板块摘要；
- `/v1/themes/{theme_id}/stocks`：板块内全部评分股票；
- `/v1/themes/{theme_id}/leaders`：板块内龙头个股；
- `/v1/candidates`：跨领先板块候选个股；
- `/v1/instruments/{instrument_id}`：个股最新价格、当日行情与最近 25 个交易日日线；
- `/v1/instruments/{instrument_id}/evidence`：个股证据与失效条件。
