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
from minigrid.core.world_object import Goal
from warehouse_env import WarehouseGridSmall, WarehouseGridTiny, WarehouseGridMedium, WarehouseGridLarge

VALID_ACTIONS = (0, 1, 2, 3, 6)  # left, right, forward, pickup, wait

ENV_DEFAULTS = {
    "WarehouseGridTiny-v0": {
        "alpha": 0.2,
        "epsilon_start": 1.0,
        "epsilon_end": 0.10,
        "epsilon_decay": 0.995,
    },
    "WarehouseGridSmall-v0": {
        "alpha": 0.18,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
    },
    "WarehouseGridMedium-v0": {
        "alpha": 0.15,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.997,
    },
    "WarehouseGridLarge-v0": {
        "alpha": 0.12,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.998,
    },
    "MiniGrid-Empty-5x5-v0": {
        "alpha": 0.1,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.999,
    },
    "MiniGrid-Empty-8x8-v0": {
        "alpha": 0.12,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.997,
    },
    "MiniGrid-LavaGapS7-v0": {
        "alpha": 0.15,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
    },
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
    parser = argparse.ArgumentParser(description="Train Q-learning on MiniGrid")
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
        help="Environment to train on",
    )
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--alpha", type=float, default=None, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=None, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=None, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=None, help="Multiplicative epsilon decay per episode")
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


def make_env(env_id: str, render_mode: str | None = None) -> gym.Env:
    return gym.make(env_id, render_mode=render_mode)


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

    # Use local blockage information to help generalize across warehouse layouts
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

def epsilon_greedy_action(
    q_table: defaultdict,
    state: tuple[int, ...],
    epsilon: float,
    rng: np.random.Generator,
    env: gym.Env,
    previous_action: int | None = None,
    repeated_turns: int = 0,
    position_loop: bool = False,
) -> int:
    valid_actions = get_valid_actions(env, previous_action)

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

    if rng.random() < epsilon:
        return int(rng.choice(valid_actions))

    q_values = q_table[state]
    valid_q_values = q_values[valid_actions]
    best_value = float(np.max(valid_q_values))
    best_actions = [
        action for action in valid_actions
        if float(q_values[action]) == best_value
    ]

    return int(rng.choice(best_actions))

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

def evaluate_policy(q_table: defaultdict, env_id: str, episodes: int = 100, seed: int = 123) -> tuple[float, float]:
    env = make_env(env_id, render_mode=None)
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    success_count = 0

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        del obs, info
        state = get_state(env)
        done = False
        total_reward = 0.0

        previous_action = None
        repeated_turns = 0
        recent_positions = []

        while not done:
            position_loop = len(recent_positions) >= 12 and len(set(recent_positions)) <= 3
            action = choose_action(q_table, state, rng, env, previous_action, repeated_turns, position_loop)

            obs, reward, terminated, truncated, info = env.step(action)

            agent_pos = tuple(env.unwrapped.agent_pos)
            recent_positions.append(agent_pos)
            if len(recent_positions) > 12:
                recent_positions.pop(0)

            if info.get("success", False):
                success_count += 1

            del obs, info

            if action in (0, 1) and previous_action == action:
                repeated_turns += 1
            elif action in (0, 1):
                repeated_turns = 1
            else:
                repeated_turns = 0

            previous_action = action

            state = get_state(env)
            total_reward += float(reward)
            done = terminated or truncated

        returns.append(total_reward)

    env.close()
    return float(np.mean(returns)), success_count / episodes


def main() -> None:
    args = parse_args()
    env_defaults = ENV_DEFAULTS.get(args.env, {})
    args.alpha = env_defaults.get("alpha", args.alpha if args.alpha is not None else 0.1)
    args.epsilon_start = env_defaults.get("epsilon_start", args.epsilon_start if args.epsilon_start is not None else 1.0)
    args.epsilon_end = env_defaults.get("epsilon_end", args.epsilon_end if args.epsilon_end is not None else 0.05)
    args.epsilon_decay = env_defaults.get("epsilon_decay", args.epsilon_decay if args.epsilon_decay is not None else 0.999)

    rng = np.random.default_rng(args.seed)
    env = make_env(args.env, render_mode=None)

    # Default each unseen state to a 7-action zero vector, matching MiniGrid's action space.
    q_table: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
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

        previous_action = None
        repeated_turns = 0
        recent_positions = []

        while not done:
            position_loop = len(recent_positions) >= 12 and len(set(recent_positions)) <= 3
            action = epsilon_greedy_action(q_table, state, epsilon, rng, env, previous_action, repeated_turns, position_loop)

            obs, reward, terminated, truncated, info = env.step(action)

            agent_pos = tuple(env.unwrapped.agent_pos)
            recent_positions.append(agent_pos)
            if len(recent_positions) > 12:
                recent_positions.pop(0)

            if info.get("success", False):
                episode_success += 1
            
            if action in (0, 1) and previous_action == action:
                repeated_turns += 1
            elif action in (0, 1):
                repeated_turns = 1
            else:
                repeated_turns = 0

            previous_action = action

            del obs, info
            next_state = get_state(env)
            done = terminated or truncated

            best_next_q = float(np.max(q_table[next_state][list(VALID_ACTIONS)]))
            td_target = float(reward) + (0.0 if done else args.gamma * best_next_q)
            td_error = td_target - float(q_table[state][action])
            q_table[state][action] += args.alpha * td_error

            state = next_state
            episode_reward += float(reward)

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

    mean_return, success_rate = evaluate_policy(q_table, args.env, episodes=100)
    print(f"\nSaved Q-table to: {args.save_path}")
    print(f"Evaluation over 100 episodes: mean_return={mean_return:.3f}, success_rate={success_rate:.2%}")

    import matplotlib.pyplot as plt
    window_size = 100
    smoothed_rewards = np.convolve(all_rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, linewidth=2, label=f'Smoothed Reward (window={window_size})')
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title(f"Smoothed Reward vs Episodes ({args.env})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
