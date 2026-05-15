#!/usr/bin/env python3
"""Load a trained Q-table and watch it act in MiniGrid."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import minigrid
from minigrid.core.world_object import Goal
from warehouse_env import WarehouseGridSmall, WarehouseGridTiny, WarehouseGridMedium, WarehouseGridLarge


VALID_ACTIONS = (0, 1, 2, 3, 6)  # left, right, forward, pickup, wait
ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    6: "wait",
}

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
    if getattr(base_env, "human_pos", None) == pos:
        return 1
    cell = base_env.grid.get(x, y)
    return int(cell is not None and cell.type == "wall")


def get_human_position(env: gym.Env) -> tuple[int, int] | None:
    return getattr(env.unwrapped, "human_pos", None)


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

    human_pos = get_human_position(env)
    if human_pos is None:
        human_dx = 0
        human_dy = 0
        human_front = 0
        human_left = 0
        human_right = 0
    else:
        human_dx = _normalize_offset(human_pos[0] - int(x))
        human_dy = _normalize_offset(human_pos[1] - int(y))
        human_front = int(front_pos == human_pos)
        human_left = int(left_pos == human_pos)
        human_right = int(right_pos == human_pos)

    return (
        direction,
        dx,
        dy,
        int(picked),
        blocked_front,
        blocked_left,
        blocked_right,
        human_dx,
        human_dy,
        human_front,
        human_left,
        human_right,
    )

def get_front_cell(env: gym.Env):
    base_env = env.unwrapped
    front_pos = tuple(base_env.front_pos)
    return front_pos, base_env.grid.get(*front_pos)


def package_is_in_front(env: gym.Env) -> bool:
    base_env = env.unwrapped
    front_pos, front_cell = get_front_cell(env)

    if getattr(base_env, "carrying", None) is not None:
        return False

    if hasattr(base_env, "package_pos"):
        return front_pos == tuple(base_env.package_pos)

    return front_cell is not None and front_cell.type == "ball"

def get_valid_actions(env: gym.Env, previous_action: int | None = None) -> list[int]:
    base_env = env.unwrapped
    front_pos, front_cell = get_front_cell(env)

    valid_actions = [0, 1]  # left, right

    front_is_wall = front_cell is not None and front_cell.type == "wall"
    front_is_human = getattr(base_env, "human_pos", None) == front_pos

    if front_is_human:
        valid_actions.append(6)  # wait only for human

    if not front_is_wall and not front_is_human:
        valid_actions.append(2)  # forward

    if package_is_in_front(env):
        valid_actions.append(3)  # pickup

    if previous_action == 0 and 1 in valid_actions and len(valid_actions) > 1:
        valid_actions.remove(1)
    elif previous_action == 1 and 0 in valid_actions and len(valid_actions) > 1:
        valid_actions.remove(0)

    return valid_actions

def choose_action(
    q_table: dict,
    state: tuple[int, ...],
    rng: np.random.Generator,
    env: gym.Env,
    previous_action: int | None = None,
    repeated_turns: int = 0,
    position_loop: bool = False,
) -> int:
    valid_actions = get_valid_actions(env, previous_action)

    # Always pick up the package when it is directly in front.
    if 3 in valid_actions:
        return 3

    if position_loop:
        escape_actions = [a for a in valid_actions if a != 6]
        return int(rng.choice(escape_actions))

    if repeated_turns >= 3:
        if 2 in valid_actions:
            return 2
        if state[9] == 1 and 6 in valid_actions:
            return 6

    q_values = q_table.get(state)
    if q_values is None:
        return int(rng.choice(valid_actions))

    valid_q_values = q_values[valid_actions]
    best_value = float(np.max(valid_q_values))
    best_actions = [
        action for action in valid_actions
        if float(q_values[action]) == best_value
    ]

    return int(rng.choice(best_actions))

def main() -> None:
    args = parse_args()
    with args.model_path.open("rb") as f:
        q_table: dict[tuple[int, ...], np.ndarray] = pickle.load(f)

    env = gym.make(args.env, render_mode=args.render_mode)

    rng = np.random.default_rng(args.seed)

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        del obs, info
        state = get_state(env)
        done = False
        total_reward = 0.0
        steps = 0

        previous_action = None
        repeated_turns = 0
        recent_positions = []

        while not done:
            position_loop = len(recent_positions) >= 12 and len(set(recent_positions)) <= 3
            action = choose_action(q_table, state, rng, env, previous_action, repeated_turns, position_loop)

            print(
                f"episode={episode} "
                f"step={steps + 1} "
                f"action={ACTION_NAMES.get(action, action)}"
            )

            obs, reward, terminated, truncated, info = env.step(action)

            agent_pos = tuple(env.unwrapped.agent_pos)
            recent_positions.append(agent_pos)
            if len(recent_positions) > 12:
                recent_positions.pop(0)

            if action in (0, 1) and previous_action == action:
                repeated_turns += 1
            elif action in (0, 1):
                repeated_turns = 1
            else:
                repeated_turns = 0

            previous_action = action

            del obs, info

            state = get_state(env)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

        print(f"episode={episode} steps={steps} total_reward={total_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
