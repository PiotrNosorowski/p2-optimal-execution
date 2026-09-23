import torch
from torch.optim import Adam
from torch.distributions import Normal
from marl_env import Execution

def update(model, optimizer, buffer, n_epochs=10, epsilon=0.2):

    # conversion to tensors
    states = torch.tensor(buffer.states)
    actions = torch.tensor(buffer.actions)
    old_log_probs = torch.tensor(buffer.log_probs)
    returns = torch.tensor(buffer.returns)
    old_values = torch.tensor(buffer.values)


    # advantages = [real return] - [predicted return]
    advantages = returns - old_values


    for epoch in range(n_epochs):
        mu_new, values_new = model(states)  

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




