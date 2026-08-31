# Worker

当前提供可复用的同步数据流水线和本地 CLI 入口：

```powershell
python -m quant_agent demo
python -m quant_agent research demo
python -m quant_agent data sync --help
python -m quant_agent data status
```

实现位于 `packages/quant_agent/pipelines/`。增量水位、租约、失败恢复、RAW-first 归档和
记录级血缘已经实现；当前仍是按次运行的 CLI，常驻调度器和跨主机分布式执行将在后续阶段实现。

