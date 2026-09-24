import torch
from torch.optim import Adam
from torch.distributions import Normal
from marl_env import Execution
import numpy as np

def update(model, optimizer, buffer, n_epochs=10, epsilon=0.2):

    # conversion to tensors
    # as UserWarning due to calculation speed occurs, lists are converted to single numpy arrays
    # Pytorch creates float64 by default, yet net requires float32
    states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
    actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32)

    # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    # ... so as old_log_probs and old_values are returned by net
    old_log_probs = torch.tensor(np.array([lp.detach().numpy() for lp in buffer.log_probs]), dtype=torch.float32).squeeze()
    returns = torch.tensor(np.array(buffer.returns), dtype=torch.float32)
    # returns normalisation
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    old_values = torch.tensor(np.array([v.detach().numpy() for v in buffer.values]), dtype=torch.float32).squeeze()

    # advantages = [real return] - [predicted return]
    advantages = returns - old_values
    # MUST BE NORMALIZED AS IT'S SCALE IS TOO BIG TO HANDLE BY GRADIENT
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for epoch in range(n_epochs):
        mu_new, values_new = model(states)
        mu_new = mu_new.squeeze()
        values_new = values_new.squeeze() 

        # as exponent cancels logarithm
        std = torch.exp(model.log_std)                 # parameter taken from model 
        dist_new = Normal(mu_new, std)                 # forming a distribution

        # new log_prob under the updated policy, for the same actions
        # ... compared to old_log_probs via ratio below
        new_log_probs = dist_new.log_prob(actions)     

        # for training improvement check
        ratio = torch.exp(new_log_probs - old_log_probs)

        # as ratio must be cut for trainig's stability purposes
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

        # 3 losses to fulfill the objective
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        critic_loss = ((returns - values_new) ** 2).mean()
        entropy = dist_new.entropy().mean()

        # the complete PPO's objective
        loss = actor_loss + critic_loss - 0.01 * entropy

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        



