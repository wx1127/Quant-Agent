# Web

P7-T06 提供一个无构建依赖的市场研究台，直接复用 API 的研究快照：

```powershell
cd apps/web
python -m http.server 5173
```

打开 `http://localhost:5173`，API 默认指向 `http://localhost:8000/v1`。页面展示市场阶段、行业主线、龙头、候选、数据时间和版本；如果需要覆盖认证令牌，可在浏览器控制台设置 `localStorage.setItem('quantAgentToken', 'user:research')`。

