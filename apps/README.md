# Applications

应用入口按运行边界拆分：

- `api/`：对外 API；
- `worker/`：数据与研究任务；
- `execution_gateway/`：模拟盘和券商执行，必须与研究进程隔离；
- `web/`：Web 控制台。

当前 `worker` 已提供本地演示、Point-in-time 研究演示、RAW-first Tushare 同步与运行状态
查询；API 已提供鉴权、研究、回测、组合、订单审批、报告和 Agent 对话接口，Web 已提供
研究、运维和对话页面。执行网关仍保持模拟/受控边界，真实券商联调不在当前版本范围内。

