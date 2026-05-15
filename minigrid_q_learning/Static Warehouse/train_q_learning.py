#!/usr/bin/env python3
"""Train a tabular Q-learning agent on a custom MiniGrid warehouse environment."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from WarehouseEnv import WarehouseEnv


ENV_NAME = "WarehouseV1-DynamicBlockage"
VALID_ACTIONS = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-learning on WarehouseEnv")

    parser.add_argument("--episodes", type=int, default=80000)

    parser.add_argument("--alpha", type=float, default=0.1)

    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--epsilon-start", type=float, default=1.0)

    parser.add_argument("--epsilon-end", type=float, default=0.02)

    parser.add_argument("--epsilon-decay", type=float, default=0.99985)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("results/q_table_warehouse.pkl"),
    )

    parser.add_argument("--log-every", type=int, default=100)

    return parser.parse_args()


def make_env(render_mode=None):
    return WarehouseEnv(render_mode=render_mode)


def get_state(env) -> tuple[int, int, int, int, int, int, int]:
    base_env = env.unwrapped

    x, y = base_env.agent_pos
    direction = int(base_env.agent_dir)

    carrying = int(base_env.carrying is not None)

    blocked_x, blocked_y = base_env.blocked_pos

    previous_action = int(base_env.previous_action)

    return (
        int(x),
        int(y),
        direction,
        carrying,
        int(blocked_x),
        int(blocked_y),
        previous_action,
    )


def epsilon_greedy_action(
    q_table: defaultdict[tuple[int, int, int, int, int, int, int], np.ndarray],
    state: tuple[int, int, int, int, int, int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:

    (
        x,
        y,
        direction,
        carrying,
        blocked_x,
        blocked_y,
        previous_action,
    ) = state

    # Always-valid movement actions
    valid_actions = [0, 1, 2]

    # Only allow pickup if not carrying
    if carrying == 0:
        valid_actions.append(3)

    # Only allow drop if carrying
    else:
        valid_actions.append(4)

    # Exploration
    if rng.random() < epsilon:
        return int(rng.choice(valid_actions))

    # Exploitation
    q_values = q_table[state]

    valid_q_values = q_values[valid_actions]

    max_q = np.max(valid_q_values)

    best_actions = [
        action
        for action, q in zip(valid_actions, valid_q_values)
        if q == max_q
    ]

    return int(rng.choice(best_actions))


def greedy_action(
    q_table: defaultdict[tuple[int, int, int, int, int, int, int], np.ndarray],
    state: tuple[int, int, int, int, int, int, int],
    rng: np.random.Generator,
) -> int:

    (
        x,
        y,
        direction,
        carrying,
        blocked_x,
        blocked_y,
        previous_action,
    ) = state

    valid_actions = [0, 1, 2]

    if carrying == 0:
        valid_actions.append(3)
    else:
        valid_actions.append(4)

    q_values = q_table[state]

    valid_q_values = q_values[valid_actions]

    max_q = np.max(valid_q_values)

    best_actions = [
        action
        for action, q in zip(valid_actions, valid_q_values)
        if q == max_q
    ]

    return int(rng.choice(best_actions))


def evaluate_policy(
    q_table: defaultdict[tuple[int, int, int, int, int, int, int], np.ndarray],
    episodes: int = 100,
    seed: int = 123,
) -> tuple[float, float]:

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

        episode_success = False

        while not done:

            action = greedy_action(q_table, state, rng)

            obs, reward, terminated, truncated, info = env.step(action)
            del obs

            state = get_state(env)

            total_reward += float(reward)

            done = terminated or truncated

            if info.get("success", False):
                episode_success = True

        if episode_success:
            success_count += 1

        returns.append(total_reward)

    env.close()

    return float(np.mean(returns)), success_count / episodes


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    env = make_env(render_mode=None)

    q_table: defaultdict[
        tuple[int, int, int, int, int, int, int],
        np.ndarray,
    ] = defaultdict(
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

            action = epsilon_greedy_action(
                q_table,
                state,
                epsilon,
                rng,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            del obs

            next_state = get_state(env)

            done = terminated or truncated

            best_next_q = float(
                np.max(q_table[next_state][list(VALID_ACTIONS)])
            )

            td_target = float(reward)

            if not done:
                td_target += args.gamma * best_next_q

            td_error = td_target - float(q_table[state][action])

            q_table[state][action] += args.alpha * td_error

            state = next_state

            episode_reward += float(reward)

            if info.get("success", False):
                episode_success = 1

        rewards_window.append(episode_reward)

        success_window.append(episode_success)

        all_rewards.append(episode_reward)

        if len(rewards_window) > args.log_every:
            rewards_window.pop(0)
            success_window.pop(0)

        epsilon = max(
            args.epsilon_end,
            epsilon * args.epsilon_decay,
        )

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

    mean_return, success_rate = evaluate_policy(
        q_table,
        episodes=100,
    )

    print(f"\nSaved Q-table to: {args.save_path}")

    print(
        f"Evaluation over 100 episodes: "
        f"mean_return={mean_return:.3f}, "
        f"success_rate={success_rate:.2%}"
    )

    import matplotlib.pyplot as plt

    window_size = 100

    if len(all_rewards) >= window_size:

        smoothed_rewards = np.convolve(
            all_rewards,
            np.ones(window_size) / window_size,
            mode="valid",
        )

        plt.plot(smoothed_rewards)

        plt.xlabel("Episodes")

        plt.ylabel("Average Reward")

        plt.title(
            f"Smoothed Reward vs Episodes ({ENV_NAME})"
        )

        plt.show()


if __name__ == "__main__":
    main()