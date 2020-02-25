import numpy as np
import pandas as pd


class MultiArmedBandit:
    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True


class GaussianBandit(MultiArmedBandit):

    def __init__(self, k, mu=0, sigma=1):
        super(GaussianBandit, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return np.random.normal(self.action_values[action]), action == self.optimal


class AdBandit(MultiArmedBandit):
    def __init__(self, data_path):
        self.ads_data = pd.read_csv(data_path)
        super(AdBandit, self).__init__(self.ads_data.shape[1])

    def reset(self):
        self.action_values = self.ads_data.sample(n=1, replace=True).values.squeeze()
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return self.action_values[action], action == self.optimal
