#!/usr/bin/env python3
"""Train a tabular Q-learning agent on MiniGrid-Empty-5x5-v0.

This example intentionally uses a compact state representation:
    (agent_x, agent_y, agent_dir)

That makes tabular Q-learning feasible on a simple MiniGrid task.
It is a good educational baseline, but it will not scale well to harder,
partially observed tasks like DoorKey without richer state handling.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import minigrid


#ENV_ID = "MiniGrid-Empty-5x5-v0"
#ENV_ID = "MiniGrid-Empty-8x8-v0"
ENV_ID = "MiniGrid-LavaGapS7-v0"
VALID_ACTIONS = (0, 1, 2)  # left, right, forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-learning on MiniGrid")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.999, help="Multiplicative epsilon decay per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("q_table.pkl"),
        help="Where to save the learned Q-table",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print training stats every N episodes",
    )
    return parser.parse_args()


def make_env(render_mode: str | None = None) -> gym.Env:
    return gym.make(ENV_ID, render_mode=render_mode)


def get_state(env: gym.Env) -> tuple[int, int, int]:
    # MiniGrid environments expose the agent pose on the unwrapped env.
    base_env = env.unwrapped
    x, y = base_env.agent_pos
    direction = int(base_env.agent_dir)
    return int(x), int(y), direction


def epsilon_greedy_action(
    q_table: defaultdict,
    state: tuple[int, int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.choice(VALID_ACTIONS))

    q_values = q_table[state]
    valid_q_values = q_values[list(VALID_ACTIONS)]
    best_action_index = int(np.argmax(valid_q_values))
    return VALID_ACTIONS[best_action_index]


def evaluate_policy(q_table: defaultdict, episodes: int = 100, seed: int = 123) -> tuple[float, float]:
    env = make_env(render_mode=None)
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    success_count = 0

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        del obs, info
        state = get_state(env)
        done = False
        total_reward = 0.0

        while not done:
            q_values = q_table[state]
            valid_q_values = q_values[list(VALID_ACTIONS)]
            best_action_index = int(np.argmax(valid_q_values))
            action = VALID_ACTIONS[best_action_index]

            obs, reward, terminated, truncated, info = env.step(action)
            del obs, info
            state = get_state(env)
            total_reward += float(reward)
            done = terminated or truncated
            if terminated and reward > 0:
                success_count += 1

        returns.append(total_reward)

    env.close()
    return float(np.mean(returns)), success_count / episodes


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = make_env(render_mode=None)

    # Default each unseen state to a 7-action zero vector, matching MiniGrid's action space.
    q_table: defaultdict[tuple[int, int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(env.action_space.n, dtype=np.float32)
    )

    epsilon = args.epsilon_start
    rewards_window: list[float] = []
    success_window: list[int] = []
    all_rewards: list[float] = []

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        del obs, info
        state = get_state(env)
        done = False
        episode_reward = 0.0
        episode_success = 0

        while not done:
            action = epsilon_greedy_action(q_table, state, epsilon, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            del obs, info
            next_state = get_state(env)
            done = terminated or truncated

            best_next_q = float(np.max(q_table[next_state][list(VALID_ACTIONS)]))
            td_target = float(reward) + (0.0 if done else args.gamma * best_next_q)
            td_error = td_target - float(q_table[state][action])
            q_table[state][action] += args.alpha * td_error

            state = next_state
            episode_reward += float(reward)
            if terminated and reward > 0:
                episode_success = 1

        rewards_window.append(episode_reward)
        success_window.append(episode_success)
        all_rewards.append(episode_reward)
        if len(rewards_window) > args.log_every:
            rewards_window.pop(0)
            success_window.pop(0)

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode % args.log_every == 0:
            print(
                f"episode={episode:5d} "
                f"epsilon={epsilon:.3f} "
                f"avg_reward={np.mean(rewards_window):.3f} "
                f"success_rate={np.mean(success_window):.2%}"
            )

    env.close()

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    with args.save_path.open("wb") as f:
        pickle.dump(dict(q_table), f)

    mean_return, success_rate = evaluate_policy(q_table, episodes=100)
    print(f"\nSaved Q-table to: {args.save_path}")
    print(f"Evaluation over 100 episodes: mean_return={mean_return:.3f}, success_rate={success_rate:.2%}")

    import matplotlib.pyplot as plt
    window_size = 100
    smoothed_rewards = np.convolve(all_rewards, np.ones(window_size)/window_size, mode='valid')

    plt.plot(smoothed_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title(f"Smoothed Reward vs Episodes ({ENV_ID})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
