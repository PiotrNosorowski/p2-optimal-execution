# g solver

from scipy.integrate import solve_ivp
from coupled_riccati import T, kappa, fun, gamma

n = 5
g=10

# count alpha and beta
sol_ricatti = solve_ivp(fun, (T, 0), [kappa, 0], dense_output=True, rtol=1e-10, atol=1e-12)

def traj(t, g):
    alpha, beta = sol_ricatti.sol(t)
    u = (1/kappa) * ((alpha - gamma)*g + ((-2*beta + gamma)  / (n + 1)) * (n * g))
    dg = -u
    return dg

# having alpha and beta, count trajectory
sol_g = solve_ivp(traj, t_span=(0, T), y0=[g], dense_output=True, rtol=1e-10, atol=1e-12)

print(sol_g.y)

