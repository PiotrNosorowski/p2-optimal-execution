# PettingZoo's Parallel API pattern must be used

from marl_env import Execution
from ppo import ActorCritic
from buffer import Buffer
import torch
from torch.optim import Adam
from update import update

env = Execution(n=5, kappa=1, gamma=0.2, varphi=0.25, T=10, g0=10, n_steps=50, A=1)

n_episodes = 1000

models = {agent: ActorCritic() for agent in env.possible_agents}
optimizers = {agent: Adam(models[agent].parameters(), lr=3e-4) for agent in env.possible_agents}
buffers = {agent: Buffer() for agent in env.possible_agents}


for episode in range(n_episodes):

    observations, infos = env.reset(seed=42)

    while env.agents:

        actions = {}
        raw_actions = {}                             # NEW: unclipped samples, for the PPO update
        log_probs = {}
        values = {}

        for agent in env.agents:

            u, u_raw, log_prob, value = models[agent].get_action(torch.tensor(observations[agent], dtype=torch.float32))   # NEW: 4 outputs

            actions[agent] = u.item()                # clipped action -> environment
            raw_actions[agent] = u_raw.item()        # NEW: unclipped action -> buffer
            log_probs[agent] = log_prob
            values[agent] = value

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in actions:
            buffers[agent].store(observations[agent], raw_actions[agent], rewards[agent], log_probs[agent], values[agent])   # NEW: raw_actions

        # the observation for the next loop to rely on
        observations = next_observations

    for agent in models:
        buffers[agent].compute_returns()
        update(models[agent], optimizers[agent], buffers[agent])
        buffers[agent].clear()


test_states = torch.tensor([[1.0, 0.0], [0.1, 0.0]], dtype=torch.float32)
with torch.no_grad():
    mu_t, val_t = models["agent_0"].forward(test_states)
print(f"final diff = {val_t[0].item() - val_t[1].item():.4f}")
    

env.close() 
torch.save(models["agent_0"].state_dict(), "agent_0_fixA_ep1000.pt")