import gym
import numpy as np
import torch
from torch import nn


class CrossEntropyAgent(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_n),
            nn.Tanh()
        )
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, input):
        return self.network(input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.network(state)
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(len(action_prob), p=action_prob)
        return action

    def update_policy(self, elite_sessions):
        elite_states, elite_actions = [], []
        for session in elite_sessions:
            elite_states.extend(session['states'])
            elite_actions.extend(session['actions'])

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        loss = self.loss(self.network(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return None


def get_session(env, agent, session_len, visual=False):
    session = {}
    states, actions = [], []
    total_reward = 0
    modified_reward = 0

    state = env.reset()
    for _ in range(session_len):
        states.append(state)
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, 3)
        else:
            action = agent.get_action(state)
        actions.append(action)

        if visual:
            env.render()

        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        modified_reward += reward + 300 * (gamma * abs(new_state[1]) - abs(state[1]))

        if done:
            break

    session['states'] = states
    session['actions'] = actions
    session['total_reward'] = total_reward
    session['modified_reward'] = modified_reward
    return session


def get_elite_sessions(sessions, q_param):

    total_rewards = np.array([session['total_reward'] for session in sessions])
    quantile = np.quantile(total_rewards, q_param)

    elite_sessions = []
    for session in sessions:
        if session['total_reward'] > quantile:
            elite_sessions.append(session)

    return elite_sessions


env = gym.make("MountainCar-v0").env
agent = CrossEntropyAgent(2, 3)

episode_n = 200
session_n = 20
session_len = 2000
q_param = 0.8
gamma = 0.7

epsilon = 1
epsilon_decay = 0.05

epsilon_min = 0.01

for episode in range(episode_n):
    epsilon = max(epsilon - epsilon_decay * episode, epsilon_min)
    sessions = [get_session(env, agent, session_len) for _ in range(session_n)]

    mean_total_reward = np.mean([session['total_reward'] for session in sessions])
    mean_modified_reward = np.mean([session['modified_reward'] for session in sessions])
    print(f'E: {episode}, mean_total_reward = {mean_total_reward}, modified reward = {mean_modified_reward}')

    elite_sessions = get_elite_sessions(sessions, q_param)

    if len(elite_sessions) > 0:
        agent.update_policy(elite_sessions)


get_session(env, agent, session_len, visual=True)