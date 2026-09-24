from marl_env import Execution
from ppo import ActorCritic
import torch
import matplotlib.pyplot as plt
from race_to_sell import count

env = Execution(n=5, kappa=1, gamma=1, varphi=0.25, T=10, g0=10, n_steps=50)

model = ActorCritic()
model.load_state_dict(torch.load("agent_0_lr3e4_ent001_ep2000.pt"))
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

plt.plot(t_analytic, g_analytic, label="analytic Nash (n=5)")
plt.plot(time_marl, trajectory, label="trained MARL agent")
plt.xlabel("step / time")
plt.ylabel("position g")
plt.legend()
plt.savefig("convergence_lr3e4_ent001_ep2000.jpg")
plt.show()




