import numpy as np
import matplotlib.pyplot as plt

kappa = 1
varphi = 0.25
gamma = 0.2
A = 1
T = 10
g0 = 10


def count(n, gamma=gamma):
    theta = np.sqrt((n - 1)**2 * gamma**2 + 16 * varphi * kappa) / (4 * kappa)
    r_plus = -(n - 1) * gamma / (4 * kappa) + theta
    r_minus = -(n - 1) * gamma / (4 * kappa) - theta
    k_T = (A - gamma / 2) / kappa          # selling speed per unit of position at T
 
    def y(t):
        return (-(r_minus + k_T) * np.exp(-r_plus * (T - t)) / (2 * theta)
                + (r_plus + k_T) * np.exp(-r_minus * (T - t)) / (2 * theta))
 
    # returns evenly separated samples
    t = np.linspace(0, T, 200)
 
    # position of every agent
    return t, g0 * y(t) / y(0)
 
 
if __name__ == "__main__":
    t1, g1 = count(1)
    t5, g5 = count(5)
 
    plt.plot(t1, g1, label="n=1 (1 player)")
    plt.plot(t5, g5, label="n=5 (5 players)")
    plt.xlabel("time t")
    plt.ylabel("position g(t)")
    plt.title("Nash equilibrium: remaining position (t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("nash_equilibrium.jpg")
    plt.show()

   