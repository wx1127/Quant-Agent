# Applications

应用入口按运行边界拆分：

- `api/`：对外 API；
- `worker/`：数据与研究任务；
- `execution_gateway/`：模拟盘和券商执行，必须与研究进程隔离；
- `web/`：Web 控制台。

P0 只建立职责边界，后续阶段再加入业务实现。

