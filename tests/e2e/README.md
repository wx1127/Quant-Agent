# P8 端到端测试

固定数据集和 Fake Broker 只运行于 `PAPER` 模式。执行
`python -m tests.e2e.runner` 可连续运行两次完整模拟交易闭环，验证结果确定性、
幂等提交和账实核对；失败诊断写入 `artifacts/e2e/failure.json`。
