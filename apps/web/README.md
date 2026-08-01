# Web

P7 提供由 FastAPI 服务的轻量 Web 工作台：

- `/web/research`：市场阶段、可点击行业主线和龙头候选个股；
- `/web/theme?theme_id=...`：板块摘要及板块内全部股票评分；
- `/web/evidence`：个股支持证据、反对证据和失效条件；
- `/web/trading`：组合风险、订单人工审批和模拟提交；
- `/web/chat`：带决策 ID 和数据截止时间的受控研究对话。

访问令牌只保存在当前浏览器标签页的 `sessionStorage`，不会写入长期
`localStorage`；关闭标签页后失效。页面不会保存券商凭证，风险警告区域不可折叠，
重复提交通过服务端幂等键和前端按钮状态共同阻止。
