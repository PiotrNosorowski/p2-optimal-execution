# PettingZoo's Parallel API pattern must be used

from marl_env import Execution
from ppo import ActorCritic
from buffer import Buffer
import torch


env = Execution(n=5, kappa=1, gamma=1, varphi=0.25, T=10, g0=10, n_steps=50)
observations, infos = env.reset(seed=42)
model = ActorCritic()
buffer = Buffer()

while env.agents:
    # requesting new action
    for agent in env.agents:
        # as observations return an array of [g,t]
        actions = model.get_action()
        u, log_prob, value = actions


        # [g,t], rewards, ...
    observations, rewards, terminations, truncations, infos = env.step({agent:u})


env.close() 