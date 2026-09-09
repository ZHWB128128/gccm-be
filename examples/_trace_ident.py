import sys
sys.path.insert(0, "/opt/boptest")
import numpy as np
from testcase import TestCase
from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import SystemState

tc = TestCase(fmupath="testcases/bestest_air/models/wrapped.fmu")
tc.step = 300.0
tc.initialize(start_time=289*86400, warmup_period=86400)
tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                 "temperature_uncertainty": None, "solar_uncertainty": None, "seed": None})
m = StateManifold(labels=["T_air","T_wall"], units={}, bounds={},
                  scale={"T_air":5.,"T_wall":5.})
e = GCCMEngine(simulator=Simulator(RCBuildingModel(), HVACModel(q_min=-15, q_max=0.)),
               manifold=m, horizon=48, dt=1/12, setpoints={"T_air":24.},
               comfort_min=22., comfort_max=27., comfort_margin=1.,
               comfort_weight=15., energy_weight=1.5, smooth_weight=1e-4,
               enforce_comfort_constraints=True, identification_enabled=True)
st = SystemState(np.array([24., 24.]), ["T_air","T_wall"])
prev = None
for k in range(120):
    dec = e.optimize(st, k/12., prev_control=prev, forced_mode="comfort")
    prev = dec.control
    u = float(dec.control.u[0])
    sp = 24. - min(1., abs(u)/15.)*(26.-18.)
    y = tc.advance({"con_oveTSetCoo_u": sp+273.15, "con_oveTSetCoo_activate":1,
                    "con_oveTSetHea_u":285.15, "con_oveTSetHea_activate":1,
                    "fcu_oveTSup_u":288.15, "fcu_oveTSup_activate":1,
                    "fcu_oveFan_u":0.6 if u<-0.01 else 0., "fcu_oveFan_activate":1})
    yp = y[2] if isinstance(y, tuple) else y
    t_true = float(yp.get("zon_reaTRooAir_y", 297.)) - 273.15
    if dec.predicted_next_state is not None:
        pe = float(np.max(np.abs(dec.predicted_next_state.x
                                 - np.array([t_true, float(st.x[1])]))))
        e.self_monitor.update(pe)
    from gccm_be.types import ExternalInput as EI
    e.observe_step(st, dec.control,
                   EI(np.array([30.0, 0.5, 0.4, 1.0]), ["T_out","solar","occ","price"]),
                   SystemState(np.array([t_true, float(st.x[1])]), ["T_air","T_wall"]), 1/12)
    st = SystemState(np.array([t_true, float(st.x[1])]), ["T_air","T_wall"])
sm = e.self_monitor.recent_mean(6)
rc = float(np.mean(np.abs(e.rc_identifier.history[-20:])))
dt = e.dt
print(f"sm={sm:.4f} rc*dt={rc*dt:.4f} gate={rc*dt < sm*0.9}")
# 手动走 apply 的影子验证段
params = e.rc_identifier.parameters()
print("params:", {k: round(v,4) if v else v for k,v in params.items()})
print("plausible:", e._rc_params_plausible(params))
c_air_c = 1.0 / params["one_over_C_air"]
print(f"候选: c_air={c_air_c:.3f} r_air={params['R_air_times_C_air']/c_air_c:.3f}")
step = e._recent_step
cand_model = RCBuildingModel(c_air=c_air_c, c_wall=4.0, r_air=params['R_air_times_C_air']/c_air_c,
                             r_wall=params['R_wall_times_C_air']/c_air_c, solar_gain=params['solar_gain_over_C_air']*c_air_c)
cand_sim = Simulator(cand_model, HVACModel(q_min=-15, q_max=0.))
pred_cand = cand_sim.step(step['state'], step['control'], step['external'], step['dt']).x[0]
pred_curr = e.simulator.step(step['state'], step['control'], step['external'], step['dt']).x[0]
actual = step['next_state'].x[0]
print(f"影子: cand={pred_cand:.3f} curr={pred_curr:.3f} actual={actual:.3f}")
print(f"err_cand={abs(pred_cand-actual):.4f} err_curr={abs(pred_curr-actual):.4f}")
print("影子通过(候选更准):", abs(pred_cand-actual) < abs(pred_curr-actual))

