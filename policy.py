import numpy as np


class Policy:

    def __str__(self):
        # return 'generic policy'
        return '(\u03B5={})'.format(0)
    def choose(self, agent):
        return 0


class EpsilonGreedyPolicy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '(\u03B5={})'.format(self.epsilon)

        # return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)


class GreedyPolicy(EpsilonGreedyPolicy):
    """
    epsilon = 0 i.e. always exploit.
    """

    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return 'greedy'


class RandomPolicy(EpsilonGreedyPolicy):
    """
    epsilon = 1 i.e. always explore.
    """

    def __init__(self):
        super(RandomPolicy, self).__init__(1)

    def __str__(self):
        return 'random'
