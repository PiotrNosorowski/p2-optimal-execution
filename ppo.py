import torch
from torch import nn
from torch.distributions import Normal 

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # common part
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)

        # actor head
        self.actor = nn.Linear(64, 1)

        # critic head
        self.critic = nn.Linear(64, 1)

        # log sigma, 1D tensor
        # must be optimized during training
        # at the beginning log_std=0 -> std=exp(0)=1
        self.log_std = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        # actor head, the mean
        # mu = self.actor(x)
        # CLAMP HAS ZERO GRADIENT BELOW ZERO -> training stucks
        mu = torch.nn.functional.softplus(self.actor(x))

        # critic head, state value
        value = self.critic(x)

        return mu, value 


    def get_action(self, x):

        mu, value = self.forward(x)          # value is passed to training buffer
        std = torch.exp(self.log_std)
        
        dist = Normal(mu, std)
        u_raw = dist.sample()

        # log probability of the action that was actually sampled (before clipping)
        log_prob = dist.log_prob(u_raw)

        # only the environment gets the clipped action; what's under 0 gets 0
        u = torch.clamp(u_raw, min=0)

        return u, u_raw, log_prob, value


