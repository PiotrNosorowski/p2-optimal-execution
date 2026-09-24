# PettingZoo's Parallel API pattern must be used

from marl_env import Execution
from ppo import ActorCritic
from buffer import Buffer
import torch
from torch.optim import Adam
from update import update

env = Execution(n=5, kappa=1, gamma=1, varphi=0.25, T=10, g0=10, n_steps=50)

n_episodes = 2000

models = {
    "agent_0": ActorCritic(),
    "agent_1": ActorCritic(),
    "agent_2": ActorCritic(),
    "agent_3": ActorCritic(),
    "agent_4": ActorCritic(),
}

optimizers = {
    "agent_0": Adam(models["agent_0"].parameters(), lr=3e-4),
    "agent_1": Adam(models["agent_1"].parameters(), lr=3e-4),
    "agent_2": Adam(models["agent_2"].parameters(), lr=3e-4),
    "agent_3": Adam(models["agent_3"].parameters(), lr=3e-4),
    "agent_4": Adam(models["agent_4"].parameters(), lr=3e-4),
}

buffers = {
    "agent_0": Buffer(),
    "agent_1": Buffer(),
    "agent_2": Buffer(),
    "agent_3": Buffer(),
    "agent_4": Buffer(),
}


for episode in range(n_episodes):

    observations, infos = env.reset(seed=42)

    while env.agents:

        actions = {}
        log_probs = {}
        values = {}

        # requesting new action
        # iteration via env as it's in charge of agents' activity
        for agent in env.agents:

            # inserting states into agentic nets
            # input of list [g,t] converted into tensor
            u, log_prob, value = models[agent].get_action(torch.tensor(observations[agent], dtype=torch.float32))

            # as u is a tensor and env.step() does not operate on tensors
            # ... a pure value must be retrieved
            actions[agent] = u.item()                # as it's the proper action
            log_probs[agent] = log_prob
            values[agent] = value


        # as for loop is done and dicts are full
        # [g,t], rewards, ...
        
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # filling buffers with new info
        for agent in env.agents:

            # storing the state BEFORE the action was taken, 
            # ... so next_observations comes after, used below
            buffers[agent].store(observations[agent], actions[agent], rewards[agent], log_probs[agent], values[agent])

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
torch.save(models["agent_0"].state_dict(), "agent_0_lr3e4_ent001_ep2000.pt")