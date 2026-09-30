from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

kappa = 1
varphi = 0.25
T = 10
g0 = 10

# rule determining the rate of change of a
def da_dt(t, a):
     return a**2 / kappa - varphi

# returns evenly separated samples
# start, stop, n_samples
ls = np.linspace(T, 0, 200)

# to solve the initial value problem
# y0 determines the initial state, which here is a terminal condition
# and a(T) = kappa at starting point
# extended by relative and absolute tolerance to tighten solver accuracy
solver = solve_ivp(da_dt, (T, 0), y0=[kappa], t_eval=ls,
                        rtol=1e-10, atol=1e-12)

if __name__ == "__main__":
    # the curve of cost-to-go per unit of squared position -> a(t)
    plt.plot(solver.t, solver.y[0])
    plt.savefig("t1.jpg")
    plt.show()

