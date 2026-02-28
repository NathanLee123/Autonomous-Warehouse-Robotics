import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=False)
print(env)

num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.05

episodes = 50000
max_steps = 100

goal_count = 0

for episode in range(episodes):

    state, info = env.reset()

    for step in range(max_steps):

        # Explore more aggressively early
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        if reward == 1:
            goal_count += 1

        # Proper Q update
        old_value = Q[state, action]
        next_max = np.max(Q[next_state])

        Q[state, action] = old_value + alpha * (
            reward + gamma * next_max - old_value
        )

        state = next_state

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("\nTimes goal reached:", goal_count)

print("\nQ-table:")
print(Q)

print("\nPolicy:")
print(np.argmax(Q, axis=1))