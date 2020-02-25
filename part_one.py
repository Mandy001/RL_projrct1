from agent import Agent
from bandit import GaussianBandit
from env import Environment
from policy import EpsilonGreedyPolicy, GreedyPolicy

n_arms = 10
n_trials = 1000
n_experiments = 2000

bandit = GaussianBandit(n_arms)

agents = [
    Agent(bandit, GreedyPolicy()),
    Agent(bandit, EpsilonGreedyPolicy(0.01)),
    Agent(bandit, EpsilonGreedyPolicy(0.1)),
    Agent(bandit, EpsilonGreedyPolicy(0.2)),
    Agent(bandit, EpsilonGreedyPolicy(0.5)),
    Agent(bandit, EpsilonGreedyPolicy(0.8)),
]
env = Environment(bandit, agents, 'Epsilon-Greedy')
scores, optimal = env.run(n_trials, n_experiments)
env.plot_results(scores, optimal)
