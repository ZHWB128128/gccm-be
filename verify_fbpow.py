
import importlib, numpy as np
M = importlib.import_module("examples.model_mismatch_experiment")
from gccm_be.physics.models import Simulator, RCBuildingModel, HVACModel
from gccm_be.types import SystemState
# show feedback_safe_control output power at hot temps with current safe_max_cooling=2.5
eng = M._make_engine(use_kinetic=True, enable_guard=True)
provider = M.SummerProvider()
w = provider.get(8.0,1)[0]
print("当前 safe_max_cooling =", eng.safe_max_cooling)
# test feedback control at high temps - what power can it output?
for T in [27.0, 28.5, 30.0, 31.5]:
    st = SystemState([T, T+0.5], ["T_air","T_wall"])
    # force using feedback (identification not trusted path)
    eng.rc_identifier = __import__("gccm_be.physics.online_id", fromlist=["RCOnlineIdentifier"])
    # simulate trust=False by setting history short
    q = eng._feedback_safe_control(st, M.STEP_H).u
    print("T=%.1f feedback_safe q = %s" % (T, q))
print("approx steady cooling needed to hold 27 at peak:", end=" ")
P=M.PLANT_PARAMS; ww=w.w
steady=(ww[0]-27)/P["r_wall"]+P["solar_gain"]*ww[1]+ww[2]
print(round(max(0.0,-steady),2), "kW")
