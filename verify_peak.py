
import importlib, numpy as np
M = importlib.import_module("examples.model_mismatch_experiment")
from gccm_be.physics.models import Simulator, RCBuildingModel, HVACModel
from gccm_be.types import SystemState
plant_sim = Simulator(RCBuildingModel(**M.PLANT_PARAMS), HVACModel(q_min=-M.Q_MAX, q_max=M.Q_MAX))
provider = M.SummerProvider()
eng = M._make_engine(use_kinetic=True, enable_guard=True)
state = SystemState([28.0,28.0], ["T_air","T_wall"])
t=0.0; prev=None; pe=0.0; ts=[]; qs=[]; modes=[]
for k in range(M.STEPS):
    dec = eng.optimize(state, t, prev_control=prev, forced_mode="comfort", prediction_error=pe)
    w = provider.get(t,1)[0]
    pred = dec.predicted_next_state.x if dec.predicted_next_state else state.x
    sb=state.copy(); state = plant_sim.step(state, dec.control, w, M.STEP_H)
    pe = float(np.max(np.abs(pred - state.x)))
    eng.self_monitor.update(pe); eng.observe_step(sb, dec.control, w, state, M.STEP_H); eng.apply_rc_identification(min_samples=30)
    ts.append(state.x[0]); qs.append(dec.control.u[0]); modes.append("deg" if eng._degraded else "nrm")
    prev=dec.control; t+=M.STEP_H
ts=np.array(ts); qs=np.array(qs)
hot = ts>27.0
print("高温时段(>27)占: %.1f%%" % (hot.mean()*100))
print("高温时段平均制冷功率: %.2f kW (Q_max=-8)" % qs[hot].mean())
print("高温时段降级占比: %.1f%%" % (np.array([m=="deg" for m in modes])[hot].mean()*100))
# 看温度最高点附近的控制
i_peak = np.argmax(ts)
print("峰值温度 %.2f 前后3步: 控制=[%s] 降级=[%s]" % (ts[i_peak],
    ",".join("%.1f"%q for q in qs[max(0,i_peak-2):i_peak+3]),
    ",".join(modes[max(0,i_peak-2):i_peak+3])))
