from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from coupled_riccati import fun, kappa, gamma, T

g0 = 10

def count(n):
    # count alpha and beta
    sol_ricatti = solve_ivp(fun, (T, 0), y0=[kappa, 0], dense_output=True, rtol=1e-10, atol=1e-12, args=(gamma, n))

    def traj(t, g):
        alpha, beta = sol_ricatti.sol(t)
        u = (1/kappa) * ((alpha - gamma)*g + ((-2*beta + gamma)  / (n + 1)) * (n * g))
        dg = -u
        return dg

    # having alpha and beta, count trajectory
    sol_g = solve_ivp(traj, t_span=(0, T), y0=[g0], dense_output=True, rtol=1e-10, atol=1e-12) 

    # x and y for plot
    return sol_g.t, sol_g.y[0]

t1, g1 = count(1)
t5, g5 = count(5)

plt.plot(t1, g1, label="n=1 (1 player)")
plt.plot(t5, g5, label="n=5 (5 players)")
plt.xlabel("time t")
plt.ylabel("position g(t)")
plt.title("Race-to-sell: remaining position (t)")
plt.legend()
plt.grid(True)
plt.savefig("race_to_sell.jpg")
plt.show()