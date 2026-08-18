# EnergyPlus / BOPTEST 接入指南

## 当前状态

- 已提供 `BuildingSimulatorAdapter` 统一接口；
- 已提供 `EnergyPlusAdapterStub` 模拟实现；
- GCCM 控制逻辑已通过适配器闭环验证。

## 接入真实 EnergyPlus 步骤

1. 安装 EnergyPlus 和 Python API：

```bash
pip install pyenergyplus
```

2. 实现真实适配器：

```python
from gccm_be.physics.building_simulator_adapter import BuildingSimulatorAdapter

class RealEnergyPlusAdapter(BuildingSimulatorAdapter):
    def __init__(self, energyplus_worker):
        self.worker = energyplus_worker
    def reset(self, initial_state):
        # 初始化 EnergyPlus 仿真
        pass
    def step(self, control, external):
        # 写控制量到 EnergyPlus，推进一个步长，读取温度
        return state
```

3. 将 `EnergyPlusAdapterStub` 替换为 `RealEnergyPlusAdapter`；
4. GCCM 控制循环无需修改。

## BOPTEST 接入

BOPTEST 提供 HTTP API，可直接在 `step` 中调用：

```python
import requests
class BOPTESTAdapter(BuildingSimulatorAdapter):
    def step(self, control, external):
        # POST /step 到 BOPTEST
        pass
```

## 注意事项

- 外部仿真器的步长需与 GCCM 控制周期一致（当前 15 分钟）；
- 测量量需映射到 `SystemState.labels`；
- 控制量需映射到 `ControlInput.labels`。
