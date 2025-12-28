import numpy as np
from dynamics import MassDamperEnv
from stable_baselines3 import SAC
from controller import PIDController
from simulate import simulate
from plot import plot_results

model = SAC.load("mass_damper_sac")

env = MassDamperEnv()
obs = env.reset()
positions_rl = []
controls_rl = []

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    positions_rl.append(env.system.x)
    controls_rl.append(action)
    if done:
        break

plot_results({"positions":positions_rl, "controls":controls_rl}, env.dt, target=1.0, trial_name="RL Policy")

best_Kp, best_Ki, best_Kd = 0.257, 0.0, 0.216
system_pid = env.system
pid = PIDController(best_Kp, best_Ki, best_Kd, dt=env.dt, u_max=env.system.u_max)
result_pid = simulate(system_pid, pid, target=1.0)
plot_results(result_pid, env.dt, target=1.0, trial_name="PID Controller")
