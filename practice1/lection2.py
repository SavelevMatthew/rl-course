import gym
import time
import numpy as np
from IPython.display import clear_output


class Agent:
    def __init__(self, states_amount, actions_amount):
        self.states_amount = states_amount
        self.actions_amount = actions_amount
        self.q_table = np.zeros((states_amount, actions_amount))
        print(f'Agent was created! q_table shape: {self.q_table.shape}')

    def get_action(self, env, current_state, epsilon):
        return env.action_space.sample() if np.random.uniform() < epsilon \
            else np.argmax(self.q_table[current_state])

    def make_step(self, state):
        return np.argmax(self.q_table[state])


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Episode: {frame['episode']}")
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        time.sleep(1)


def on_execute():
    env = gym.make('Taxi-v3').env
    agent = Agent(env.observation_space.n, env.action_space.n)
    print('*' * 64)
    alpha, gamma, epsilon, episodes, steps = .3, .6, .1, 10000, 100
    print(f'Training with parameters: \n\talpha: {alpha}, \n\tgamma: {gamma},'
          f' \n\tepsilon: {epsilon}')
    print(f'Total episodes: {episodes}, steps allowed: {steps}')
    print('*' * 64)

    # Training
    for ep in range(episodes):
        state = env.reset()
        epoch, penalties, rewards, step, done = 0, 0, 0, 0, False
        while step < steps and not done:
            action = agent.get_action(env, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            old_value = agent.q_table[state, action]
            next_max = np.max(agent.q_table[next_state])

            # Usual update
            # new_value = reward + gamma * next_max

            # Policy сглаживание
            new_value = (1 - alpha) * old_value + alpha * (
                        reward + gamma * next_max)

            agent.q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            step += 1

        if ep % 500 == 0:
            clear_output(wait=True)
            print(f'Episode: {ep}, steps: {step}, done: {done}')

    print('*' * 64)

    # Evaluation on Test Runs
    total_epochs, total_penalties, total_reward, total_done = 0, 0, 0, 0
    frames = []
    episodes, max_steps = 100, 20
    print(f'Evaluating agent on {episodes} episodes,'
          f' with max step of {max_steps}')

    for ep in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.make_step(state)
            state, reward, done, info = env.step(action)
            total_reward += reward

            if reward == -10:
                penalties += 1

            # Put each rendered frame into dict for animation
            frames.append({
                'frame': env.render(mode='ansi'),
                'episode': ep,
                'state': state,
                'action': action,
                'reward': reward
            }
            )
            epochs += 1
            step += 1

        total_penalties += penalties
        total_epochs += epochs
        if done:
            total_done += 1

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    print(f"Average reward per episode: {total_reward / episodes}")
    print(f'Total finished tasks: {total_done}')
    answer = input('Would you like to print frames? [y/n]: ').lower()
    if answer.startswith('y'):
        print_frames(frames)


if __name__ == '__main__':
    on_execute()
