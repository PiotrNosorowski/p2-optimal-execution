from pettingzoo.test import parallel_api_test
from marl_env import Execution

env = Execution(n=5, kappa=1, gamma=1, varphi=0.25, T=10, g0=10, n_steps=50)
parallel_api_test(env, num_cycles=1000)

print("passed")