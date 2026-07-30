# Applications

应用入口按运行边界拆分：

- `api/`：对外 API；
- `worker/`：数据与研究任务；
- `execution_gateway/`：模拟盘和券商执行，必须与研究进程隔离；
- `web/`：Web 控制台。

P7 已实现 `api/` 与 `web/`；`worker/` 和真实 `execution_gateway/` 仍保留为后续
部署与实盘准备边界。当前 API 的订单提交仅连接模拟服务，不包含真实券商凭证。
