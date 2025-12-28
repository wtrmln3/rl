import numpy as np

def compute_cost(result, target=1.0, dt=0.01):
    positions = np.array(result["positions"])
    controls = np.array(result["controls"])

    e_final = abs(target - positions[-1])
    overshoot = max(0.0, positions.max() - target)

    within_tol = np.where(np.abs(positions - target) <= 0.01)[0]
    settling_time = within_tol[0] * dt if len(within_tol) > 0 else len(positions) * dt

    effort = np.sum(controls**2) * dt

    J = 5 * overshoot + 1.0 * settling_time + 0.1 * effort + 10 * e_final
    return J
