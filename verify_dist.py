
import importlib, numpy as np
M = importlib.import_module("examples.model_mismatch_experiment")
from gccm_be.physics.models import Simulator, RCBuildingModel, HVACModel
from gccm_be.types import SystemState
plant_sim = Simulator(RCBuildingModel(**M.PLANT_PARAMS), HVACModel(q_min=-M.Q_MAX, q_max=M.Q_MAX))
provider = M.SummerProvider()
eng = M._make_engine(use_kinetic=True, enable_guard=True)
state = SystemState([28.0,28.0], ["T_air","T_wall"])
t=0.0; prev=None; pe=0.0; ts=[]
for k in range(M.STEPS):
    dec = eng.optimize(state, t, prev_control=prev, forced_mode="comfort", prediction_error=pe)
    w = provider.get(t,1)[0]
    pred = dec.predicted_next_state.x if dec.predicted_next_state else state.x
    sb=state.copy(); state = plant_sim.step(state, dec.control, w, M.STEP_H)
    pe = float(np.max(np.abs(pred - state.x)))
    eng.self_monitor.update(pe); eng.observe_step(sb, dec.control, w, state, M.STEP_H); eng.apply_rc_identification(min_samples=30)
    ts.append(state.x[0]); prev=dec.control; t+=M.STEP_H
ts=np.array(ts)
hot = np.mean(ts>M.COMFORT_MAX)*100
cold = np.mean(ts<M.COMFORT_MIN)*100
print("违规分解: 高温>27: %.1f%%  低温<25: %.1f%%" % (hot, cold))
print("温度分位: p10=%.2f p50=%.2f p90=%.2f" % (np.percentile(ts,10), np.percentile(ts,50), np.percentile(ts,90)))
# count in comfort band
print("in-band(25-27): %.1f%%" % (np.mean((ts>=25)&(ts<=27))*100))
