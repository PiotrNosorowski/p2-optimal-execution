import numpy as np
from scipy.integrate import solve_ivp
from nash_et import count, kappa, T, A, g0
from riccati import da_dt

# test: one agent without permanent impact (n = 1, gamma = 0)
# must give the single-agent Riccati of riccati.py, with a(T) = A

# a(t) from riccati.py, integrated backward from a(T) = A
sol_a = solve_ivp(da_dt, (T, 0), y0=[A], dense_output=True, rtol=1e-12, atol=1e-12)

# u* = (a / kappa) * g,  dg/dt = -u*, integrated forward from g(0) = g0
sol_g = solve_ivp(lambda t, g: -(sol_a.sol(t)[0] / kappa) * g, (0, T), y0=[g0],
                  dense_output=True, rtol=1e-12, atol=1e-12)

# closed form for n = 1, gamma = 0
t1, g1 = count(1, gamma=0)

print("test: n=1, gamma=0 vs single-agent Riccati:", np.max(np.abs(sol_g.sol(t1)[0] - g1)))