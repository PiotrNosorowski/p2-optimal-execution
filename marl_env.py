from pettingzoo import ParallelEnv
from copy import copy
import numpy as np
from gymnasium.spaces import Box

class Execution(ParallelEnv):

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, n, kappa, gamma, varphi, T, g0, n_steps):
        self.n = n
        self.kappa = kappa
        self.gamma = gamma
        self.varphi = varphi
        self.T = T
        self.g0 = g0
        self.n_steps = n_steps
        self.dt = T / n_steps              # time step length
        self.possible_agents = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"]

        # as [u] belongs to the (0, g0) interval 
        self.action_spaces = {agent: Box(low=0, high=self.g0, shape=(1,)) for agent in self.possible_agents}

        # [g, timestep]
        # continuous and discrete values mixed, though Box() supports both types
        # ...and discrete values fits within continuous anyway 
        self.observation_spaces = {agent: Box(low=0, high=np.array([self.g0, self.n_steps]), shape=(2,)) for agent in self.possible_agents}
    

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # agents' initial positions
        self.positions = {agent: self.g0 for agent in self.agents}  

        # the state of the agents                                    
        observations = {agent: np.array([float(self.positions[agent]), float(self.timestep)]) for agent in self.agents}

        # API's purposes, remains empty   
        infos = {agent: {} for agent in self.agents}                     

        return observations, infos
        

    def step(self, actions):
        # .values to extract dictionary values; here: selling rates u
        total_u = sum(actions.values())
        rewards = {agent: -(self.kappa * actions[agent] * total_u + self.gamma * self.positions[agent] * total_u + self.varphi * self.positions[agent] ** 2) 
                            for agent in self.agents} 

        self.positions = {agent: (self.positions[agent] - actions[agent] * self.dt) for agent in self.agents}

        # timestep = 1/n_steps
        self.timestep += 1                # steps counter

        end = self.timestep >= self.n_steps 
        truncations = {agent: end for agent in self.agents}      # given time limit exceeded (position may still remain open)

        # as position never reaches zero -> no specific game termination exists, so it's never fully solved
        # ... so False is the setting
        terminations = {agent: False for agent in self.agents}    # game's end

        # [g,t]
        # float type added to prevent array doubling (1st term)
        # float type added for int data type (2nd term)
        # ... as Box() required homogenous array type
        observations = {agent: np.array([float(self.positions[agent]), float(self.timestep)]) for agent in self.agents}

        # unused
        infos = {agent: {} for agent in self.agents}

        # if game is ended, no active players remain 
        if end: 
            self.agents = []     # an empty list terminates the loop

        # 5 dicts to return
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]