# Applications

应用入口按运行边界拆分：

- `api/`：对外 API；
- `worker/`：数据与研究任务；
- `execution_gateway/`：模拟盘和券商执行，必须与研究进程隔离；
- `web/`：Web 控制台。

当前 `worker` 已提供本地演示、Point-in-time 研究演示、RAW-first Tushare 同步与运行状态
查询；API、执行网关和 Web 仍只建立职责边界，后续阶段再加入业务实现。

