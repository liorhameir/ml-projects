import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayMemory(object):

    def __init__(self, capacity, input_dims):
        self.capacity = capacity
        self.observations_memory = np.zeros((self.capacity, input_dims), dtype=np.float32)
        self.next_observations_memory = np.zeros((self.capacity, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.capacity, dtype=np.int64)
        self.reward_memory = np.zeros(self.capacity, dtype=np.float32)
        self.done_memory = np.zeros(self.capacity, dtype=np.bool)
        self.position = 0
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def push(self, observation, action, next_observation, reward, done):
        index = self.position % self.capacity
        self.observations_memory[index] = observation
        self.action_memory[index] = action
        self.next_observations_memory[index] = next_observation
        self.reward_memory[index] = reward
        self.done_memory[index] = done
        self.position += 1

    def extract_tensors(self, index):
        pass

    def sample(self, batch_size):
        index = len(self)
        sample = np.random.choice(index, batch_size, replace=False)
        observations = T.tensor(self.observations_memory[sample]).to(self.device)
        actions = T.from_numpy(self.action_memory[sample])
        follow_observations = T.tensor(self.next_observations_memory[sample]).to(self.device)
        rewards = T.tensor(self.reward_memory[sample]).to(self.device)
        dones = T.tensor(self.done_memory[sample]).to(self.device)
        return observations, actions, follow_observations, rewards, dones

    def __len__(self):
        return self.position if self.position < self.capacity else self.capacity


class DQN(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, t):
        x = F.relu(self.fc1(t))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, input_dims, n_actions, gamma, epsilon, lr,  batch_size,  max_mem_size=100000, epsilon_bar=1e-2,
                 eps_dec=5e-5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = epsilon_bar
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.update_target_counter = 0
        self.replace_target = 10
        self.memory = ReplayMemory(100000, input_dims)

        self.Q_policy = DQN(input_dims=input_dims, fc1_dims=80, fc2_dims=80, n_actions=n_actions)
        self.Q_target = DQN(input_dims=input_dims, fc1_dims=80, fc2_dims=80, n_actions=n_actions)
        self.Q_target.load_state_dict(self.Q_policy.state_dict())
        self.Q_target.eval()
        self.optimizer = optim.Adam(self.Q_policy.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    """
    greedy epsilon policy
    """
    def choose_action(self, observation, task='train'):
        if np.random.random() > self.epsilon or task == 'test':
            # greedy epsilon policy
            with T.no_grad():
                # expand dimensions by one so it'll fit the DQN
                actions = self.Q_policy(T.tensor(observation, dtype=T.float32).unsqueeze(dim=0))
            # pick greatest action
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        # not enough transitions in memory, so it can't sample a batch
        if len(self.memory) < self.batch_size:
            return

        self.optimizer.zero_grad()
        # extract sample
        observation, actions, next_observation, rewards, done = self.memory.sample(self.batch_size)
        # evaluate q(s,a) and get only the values related to the action.
        current_q_values = self.Q_policy(observation).gather(dim=1, index=actions.unsqueeze(-1))
        # evaluate q*(s', a')
        next_optimal_values = self.Q_target(next_observation).max(dim=1)[0]
        # q*(s, a) = R + g x q*(s', a')
        target_q_values = rewards + self.gamma * next_optimal_values
        # evaluate (q(s,a) -  q*(s, a))^2
        loss = self.loss(current_q_values.squeeze(dim=1), target_q_values).to(self.Q_policy.device)
        loss.backward()
        self.optimizer.step()
        self.update_target_counter += 1
        # epsilon greedy
        self.update_epsilon()
        if self.update_target_counter % self.replace_target == 0:
            self.Q_target.load_state_dict(self.Q_policy.state_dict())
