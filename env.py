import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Environment(object):
    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials=100, experiments=1):
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)

        for _ in range(experiments):
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    scores[t, i] += reward
                    if is_optimal:
                        optimal[t, i] += 1

        return scores / experiments, optimal / experiments

    # def plot_results(self, scores, optimal):
    #     sns.set_style('white')
    #     sns.set_context('talk')
    #     plt.subplot(2, 1, 1)
    #     plt.title(self.label)
    #     plt.plot(scores)
    #     plt.ylabel('Average Reward')
    #     plt.legend(self.agents, loc=4)
    #     plt.subplot(2, 1, 2)
    #     plt.plot(optimal * 100)
    #     plt.ylim(0, 100)
    #     plt.ylabel('% Optimal Action')
    #     plt.xlabel('Time Step')
    #     plt.legend(self.agents, loc=4)
    #     sns.despine()
    #     plt.show()


    def plot_results(self, scores, optimal):
        sns.set_style('white')
        sns.set_context('talk')
        # plt.subplot(2, 1, 1)
        plt.figure()
        plt.subplots_adjust(left=0.15, bottom=0.16)
        # plt.title(self.label)
        plt.plot(scores)
        plt.ylabel('Average Reward')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        # plt.legend(self.agents, bbox_to_anchor=(0.40, 0.8),loc='upper center')
        # plt.show()
        plt.savefig('1_1.png')
        # plt.savefig('2_1.png')
        # plt.subplot(2, 1, 2)
        plt.figure()
        # plt.title(self.label)
        plt.subplots_adjust(left=0.15, bottom=0.16)
        plt.plot(optimal * 100)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        sns.despine()
        # plt.show()
        plt.savefig('1_1.png')
        # plt.savefig('2_2.png')


