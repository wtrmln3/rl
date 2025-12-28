from skopt import gp_minimize
from skopt.space import Real
from dynamics import MassDamperSystem
from controller import PIDController
from simulate import simulate
from cost import compute_cost
from plot import plot_results
import numpy as np

def objective(params, n_trials=5):
    Kp, Ki, Kd = params
    total_cost = 0.0
    for _ in range(n_trials):
        system = MassDamperSystem(
            m=np.random.uniform(0.8,1.2),
            b=np.random.uniform(0.15,0.3),
            dt=0.01, u_max=2.0,
            noise_std=np.random.uniform(0.0,0.01)
        )
        pid = PIDController(Kp, Ki, Kd, dt=system.dt, u_max=system.u_max)
        result = simulate(system, pid, target=1.0)
        total_cost += compute_cost(result, target=1.0, dt=system.dt)
    return total_cost / n_trials

if __name__ == "__main__":
    space = [Real(0.1,10.0,'Kp'), Real(0.0,2.0,'Ki'), Real(0.0,2.0,'Kd')]
    res = gp_minimize(func=objective, dimensions=space, n_calls=70, n_random_starts=10, random_state=42)

    best_Kp, best_Ki, best_Kd = res.x
    print(f"Robust PID gains: Kp={best_Kp:.3f}, Ki={best_Ki:.3f}, Kd={best_Kd:.3f}")
    print(f"Average cost: {res.fun:.3f}")

    for i in range(5):
        system = MassDamperSystem(
            m=np.random.uniform(0.8,1.2),
            b=np.random.uniform(0.15,0.3),
            dt=0.01, u_max=2.0,
            noise_std=np.random.uniform(0.0,0.01)
        )
        pid = PIDController(best_Kp, best_Ki, best_Kd, dt=system.dt, u_max=system.u_max)
        result = simulate(system, pid, target=1.0)
        plot_results(result, system.dt, target=1.0, trial_name=f"Robust Trial {i+1}")
