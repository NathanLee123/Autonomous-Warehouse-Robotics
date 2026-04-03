#!/usr/bin/env python3
"""Load a trained Q-table and watch it act in MiniGrid."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import minigrid


#ENV_ID = "MiniGrid-Empty-5x5-v0"
#ENV_ID = "MiniGrid-Empty-8x8-v0"
ENV_ID = "MiniGrid-LavaGapS7-v0"
VALID_ACTIONS = (0, 1, 2)  # left, right, forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-learning policy")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("q_table.pkl"),
        help="Path to the saved Q-table produced by train_q_learning.py",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to render")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed for evaluation")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="MiniGrid render mode",
    )
    return parser.parse_args()


def get_state(env: gym.Env) -> tuple[int, int, int]:
    base_env = env.unwrapped
    x, y = base_env.agent_pos
    direction = int(base_env.agent_dir)
    return int(x), int(y), direction


def choose_action(q_table: dict, state: tuple[int, int, int]) -> int:
    q_values = q_table.get(state)
    if q_values is None:
        return 2  # sensible fallback: move forward

    valid_q_values = q_values[list(VALID_ACTIONS)]
    best_action_index = int(np.argmax(valid_q_values))
    return VALID_ACTIONS[best_action_index]


def main() -> None:
    args = parse_args()
    with args.model_path.open("rb") as f:
        q_table: dict[tuple[int, int, int], np.ndarray] = pickle.load(f)

    env = gym.make(ENV_ID, render_mode=args.render_mode)

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        del obs, info
        state = get_state(env)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = choose_action(q_table, state)
            obs, reward, terminated, truncated, info = env.step(action)
            del obs, info
            state = get_state(env)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

        print(f"episode={episode} steps={steps} total_reward={total_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
