import matplotlib.pyplot as plt
import numpy as np
from dynamics import MassDamperSystem
from controller import PIDController
from simulate import simulate

best_Kp, best_Ki, best_Kd = 0.257, 0.0, 0.216
target = 1.0
n_trials = 5

all_positions = []

for _ in range(n_trials):
    system = MassDamperSystem(
        m=np.random.uniform(0.8, 1.2),
        b=np.random.uniform(0.15, 0.3),
        dt=0.01,
        u_max=2.0,
        noise_std=np.random.uniform(0.0, 0.01)
    )
    pid = PIDController(best_Kp, best_Ki, best_Kd, dt=system.dt, u_max=system.u_max)
    result = simulate(system, pid, target=target)
    all_positions.append(result["positions"])

max_len = max(len(traj) for traj in all_positions)
for i in range(len(all_positions)):
    if len(all_positions[i]) < max_len:
        all_positions[i] += [all_positions[i][-1]] * (max_len - len(all_positions[i]))

all_positions = np.array(all_positions)
t = np.arange(max_len) * system.dt

pos_min = all_positions.min(axis=0)
pos_max = all_positions.max(axis=0)
pos_mean = all_positions.mean(axis=0)

plt.figure(figsize=(10,5))
plt.fill_between(t, pos_min, pos_max, color='skyblue', alpha=0.4, label="Range of Trials")
plt.plot(t, pos_mean, color='blue', label="Mean Position")
plt.axhline(target, linestyle='--', color='red', label="Target")
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.title("Robust PID Response - Shaded Envelope")
plt.legend()
plt.grid(True)
plt.show()
