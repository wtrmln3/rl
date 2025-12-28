def simulate(system, controller, target=1.0, max_steps=1000):
    system.reset()
    controller.reset()
    positions = []
    controls = []

    for _ in range(max_steps):
        u = controller.compute(target, system.x)
        x, _ = system.step(u)
        positions.append(x)
        controls.append(u)
        if abs(target - x) < 0.001:
            break

    return {"positions": positions, "controls": controls}
