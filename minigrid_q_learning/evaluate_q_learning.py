#!/usr/bin/env python3
"""Load and evaluate a trained tabular Q-learning policy on MiniGrid and warehouse environments.

This script:
    - loads a saved Q-table produced by train_q_learning.py
    - reconstructs the same handcrafted state representation used during training
    - runs evaluation episodes using the learned policy
    - renders the environment during execution
    - reports episode reward and step statistics

Supported environments include:
    - MiniGrid empty and lava tasks
    - custom WarehouseGrid environments of varying sizes

If the policy encounters an unseen state during evaluation, the agent falls
back to selecting a random valid action.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import minigrid
from minigrid.core.world_object import Goal
from warehouse_env import WarehouseGridSmall, WarehouseGridTiny, WarehouseGridMedium, WarehouseGridLarge


VALID_ACTIONS = (0, 1, 2, 3)  # left, right, forward, pickup

# Register custom warehouse environments
gym.register(
    id="WarehouseGridTiny-v0",
    entry_point="warehouse_env:WarehouseGridTiny",
)
gym.register(
    id="WarehouseGridSmall-v0",
    entry_point="warehouse_env:WarehouseGridSmall",
)
gym.register(
    id="WarehouseGridMedium-v0",
    entry_point="warehouse_env:WarehouseGridMedium",
)
gym.register(
    id="WarehouseGridLarge-v0",
    entry_point="warehouse_env:WarehouseGridLarge",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-learning policy")
    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-Empty-5x5-v0",
        choices=[
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-Empty-8x8-v0",
            "MiniGrid-LavaGapS7-v0",
            "WarehouseGridTiny-v0",
            "WarehouseGridSmall-v0",
            "WarehouseGridMedium-v0",
            "WarehouseGridLarge-v0",
        ],
        help="Environment to evaluate on",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("q_table.pkl"),
        help="Path to the saved Q-table produced by train_q_learning.py",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to render")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed for evaluation")
    parser.add_argument("--render-mode", type=str, default="human", help="Render mode (e.g. 'human', 'rgb_array')")
    return parser.parse_args()


def get_goal_position(env: gym.Env) -> tuple[int, int]:
    base_env = env.unwrapped
    for gx in range(base_env.width):
        for gy in range(base_env.height):
            cell = base_env.grid.get(gx, gy)
            if isinstance(cell, Goal):
                return int(gx), int(gy)
    raise RuntimeError("No goal found in environment")


def get_package_position(env: gym.Env) -> tuple[int, int]:
    base_env = env.unwrapped
    if getattr(base_env, "carrying", None) is not None:
        return -1, -1

    if hasattr(base_env, "package_pos"):
        return tuple(base_env.package_pos)

    for gx in range(base_env.width):
        for gy in range(base_env.height):
            cell = base_env.grid.get(gx, gy)
            if cell is not None and cell.type == "ball":
                return int(gx), int(gy)

    return -1, -1


def get_package_picked(env: gym.Env) -> int:
    base_env = env.unwrapped
    return int(getattr(base_env, "carrying", None) is not None)


def _normalize_offset(value: int, limit: int = 5) -> int:
    if value == 0:
        return 0
    return int(np.sign(value) * min(abs(value), limit))


def _is_blocked(base_env: gym.Env, pos: tuple[int, int]) -> int:
    x, y = pos
    if x < 0 or x >= base_env.width or y < 0 or y >= base_env.height:
        return 1
    cell = base_env.grid.get(x, y)
    return int(cell is not None and cell.type == "wall")


def get_state(env: gym.Env) -> tuple[int, ...]:
    base_env = env.unwrapped
    x, y = base_env.agent_pos
    direction = int(base_env.agent_dir)
    picked = get_package_picked(env)

    if picked:
        target_x, target_y = get_goal_position(env)
    else:
        target_x, target_y = get_package_position(env)

    dx = _normalize_offset(target_x - int(x))
    dy = _normalize_offset(target_y - int(y))

    front_pos = tuple(base_env.front_pos)
    left_dir = (direction - 1) % 4
    right_dir = (direction + 1) % 4
    left_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)][left_dir]
    right_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)][right_dir]
    left_pos = (int(x) + left_vec[0], int(y) + left_vec[1])
    right_pos = (int(x) + right_vec[0], int(y) + right_vec[1])

    blocked_front = _is_blocked(base_env, front_pos)
    blocked_left = _is_blocked(base_env, left_pos)
    blocked_right = _is_blocked(base_env, right_pos)

    return direction, dx, dy, int(picked), blocked_front, blocked_left, blocked_right


def choose_action(q_table: dict, state: tuple[int, int, int], rng: np.random.Generator) -> int:
    q_values = q_table.get(state)
    if q_values is None:
        # If the trained policy never saw this exact state, avoid always
        # moving forward and instead pick among valid actions.
        return int(rng.choice(VALID_ACTIONS))

    valid_q_values = q_values[list(VALID_ACTIONS)]
    best_value = float(np.max(valid_q_values))
    best_actions = [VALID_ACTIONS[i] for i, q in enumerate(valid_q_values) if q == best_value]
    return int(rng.choice(best_actions))


def main() -> None:
    args = parse_args()
    with args.model_path.open("rb") as f:
        q_table: dict[tuple[int, int, int], np.ndarray] = pickle.load(f)

    env = gym.make(args.env, render_mode=args.render_mode)

    rng = np.random.default_rng(args.seed)

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        del obs, info
        state = get_state(env)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = choose_action(q_table, state, rng)
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
