from marl_env import Execution
from ppo import ActorCritic
import torch
import matplotlib.pyplot as plt
from nash_et import count
import numpy as np

env = Execution(n=5, kappa=1, gamma=0.2, varphi=0.25, T=10, g0=10, n_steps=50, A=1)

model = ActorCritic()
model.load_state_dict(torch.load("agent_0_fixA_ep1000.pt"))
model.eval()


observations, infos = env.reset(seed=42)
trajectory = []

while env.agents:
    actions = {}
    for agent in env.agents:
        # as only u is needed here
        # u, _, _ = model.get_action(torch.tensor(observations[agent], dtype=torch.float32))
        # actions[agent] = u.item()
        # MEAN MU IS BETTER THAN U, AS NO RANDOM NOISE OCCURS
        mu, value = model.forward(torch.tensor(observations[agent], dtype=torch.float32))
        actions[agent] = mu.item()



    trajectory.append(observations["agent_0"][0] * env.g0)   # position g, rescaled from normalized [0,1] back to real units
    
    observations, rewards, terminations, truncations, infos = env.step(actions)


t_analytic, g_analytic = count(5)

# as analytical and marl axe must be comparable 
time_marl = [i * env.dt for i in range(len(trajectory))]

plt.plot(t_analytic, g_analytic, label="analytic Nash (n=1, gamma=0)")
plt.plot(time_marl, trajectory, label="trained ppo agent")
plt.xlabel("step / time")
plt.ylabel("position g")
plt.legend()
plt.savefig("agent_0_fixA_ep1000.jpg")
plt.show()


g_rl = trajectory[10]
g_benchmark = np.interp(2, t_analytic, g_analytic)
print(f"t = 2: RL {g_rl:.3f}, benchmark {g_benchmark:.3f}, gap {g_rl - g_benchmark:.3f}")


