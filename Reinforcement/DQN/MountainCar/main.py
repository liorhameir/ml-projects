import gym
from Qlearning import Agent
import numpy as np
import torch as T
import matplotlib.pyplot as plt


def plot_results(x_val, y_val, running_average, win_min):
    ax = plt.subplot()
    ax.scatter(x_val, y_val, color='green')
    ax.plot(x_val, running_average, c="red")
    ax.plot(x_val, [win_min] * len(x_val), c="yellow")
    ax.set_xlabel('Games')
    ax.set_ylabel('Scores')
    ax.set_ylim(ymin=0)
    plt.show()


def adjust_reward(next_obs):
    extra = 0
    if next_obs[0] >= 0.5:
        extra += 100
    return abs(next_obs[0] - (-0.5)) + extra


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.n, epsilon_bar=1e-2,
                  input_dims=2, lr=1e-3)

    for task in ['train', 'test']:
        print("---------" + task + "---------")
        rewards_per_game = []
        average_reward = []
        n_games = 450 if task == 'train' else 100
        if task == 'test':
            agent.Q_policy.eval()
        wins = np.zeros(n_games)
        min_win = 100
        for game in range(n_games):
            score = 0
            done = False
            observation = env.reset()
            while not done:
                # epsilon greedy policy
                action = agent.choose_action(observation, task)
                next_observation, _, done, _ = env.step(action)
                if task == 'test':
                    env.render()
                reward = adjust_reward(next_observation)
                score += reward
                if task == 'train':
                    reward = T.tensor([reward], device=agent.Q_policy.device)
                    agent.memory.push(observation, action, next_observation, reward, done)
                    agent.learn()
                observation = next_observation
            if observation[0] >= 0.5:
                wins[game] = 1
                if min_win > score:
                    min_win = score
            rewards_per_game.append(score)
            if task == 'train':
                index = max(0, game - 100)
            else:
                index = 0
            avg_score = np.mean(rewards_per_game[index:])
            average_reward.append(avg_score)
            success = np.mean(wins[index:(game + 1)]) * 100
            print('round ', game, 'score %.2f' % score, 'average score %.2f' % avg_score,
                  'wins %.1f ' % success, flush=True)
        x = [i for i in range(1, n_games + 1)]
        plot_results(x, rewards_per_game, average_reward, min_win)
    env.close()
