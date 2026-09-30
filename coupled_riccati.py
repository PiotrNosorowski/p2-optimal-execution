from scipy.integrate import solve_ivp
import numpy as np
from riccati import solver

kappa = 1
varphi = 0.25
n = 5                # n=1 for sanity check purposes
T = 10
gamma = 1


def fun(t, y, gamma, n):
    alpha = y[0]
    beta = y[1]

    dalpha = alpha * (alpha - gamma) / kappa - varphi
    dbeta = 1 / kappa * (2 * alpha * beta - (2 * n * beta + gamma) ** 2 / (n + 1) ** 2)

    return dalpha, dbeta

# returns evenly separated samples
# start, stop, n_samples
ls = np.linspace(T, 0, 200)

# extended by relative and absolute tolerance to tighten solver accuracy
# fun() extended by additional arguments args 
# intorducing gamma and n via args allows setting their values independently from those globally stated
solve_ivp_c = solve_ivp(fun, t_span=(T, 0), y0=[kappa, 0], t_eval=ls,
                        rtol=1e-10, atol=1e-12, args=(gamma, n))


if __name__ == "__main__": 
    print(np.max(np.abs(solve_ivp_c.y[0] - solver.y[0])))

    # printing beta
    print(solve_ivp_c.y[1])



