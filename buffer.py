class Buffer():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)


    def compute_returns(self):
        self.returns = []
        running_return = 0

        # the earlier step, the bigger the sum
        for reward in reversed(self.rewards):
            running_return = reward + running_return
            self.returns.append(running_return)

        # as iteration went backwards, it must be reversed now
        # ... as it will be compared against values 
        self.returns.reverse()


    # clearing lists for new episode
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.returns = []

