#!/usr/bin/env python3
"""Load a trained Q-table and watch it act in the custom warehouse environment."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from WarehouseEnv import WarehouseEnv


VALID_ACTIONS = (0, 1, 2, 3, 4)

ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained warehouse Q-learning policy"
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("results/q_table_warehouse.pkl"),
        help="Path to the saved Q-table produced by train_q_learning.py",
    )

    parser.add_argument("--episodes", type=int, default=5)

    parser.add_argument("--seed", type=int, default=1000)

    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
    )

    parser.add_argument("--delay", type=float, default=0.15)

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


def choose_action(
    q_table: dict[
        tuple[int, int, int, int, int, int, int],
        np.ndarray,
    ],
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

    # Always-valid movement actions
    valid_actions = [0, 1, 2]

    # Only allow pickup if not carrying
    if carrying == 0:
        valid_actions.append(3)

    # Only allow drop if carrying
    else:
        valid_actions.append(4)

    q_values = q_table.get(state)

    if q_values is None:
        return int(rng.choice(valid_actions))

    valid_q_values = q_values[valid_actions]

    max_q = np.max(valid_q_values)

    best_actions = [
        action
        for action, q in zip(valid_actions, valid_q_values)
        if q == max_q
    ]

    return int(rng.choice(best_actions))


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    with args.model_path.open("rb") as f:
        q_table: dict[
            tuple[int, int, int, int, int, int, int],
            np.ndarray,
        ] = pickle.load(f)

    env = make_env(render_mode=args.render_mode)

    for episode in range(1, args.episodes + 1):

        obs, info = env.reset(seed=args.seed + episode)
        del obs, info

        state = get_state(env)

        done = False

        total_reward = 0.0

        steps = 0

        success = False

        print(f"\nEpisode {episode}")

        print(
            f"blocked_pos={env.unwrapped.blocked_pos}"
        )

        while not done:

            action = choose_action(
                q_table,
                state,
                rng,
            )

            print(
                f"step={steps:03d} "
                f"state={state} "
                f"action={ACTION_NAMES[action]}"
            )

            obs, reward, terminated, truncated, info = env.step(action)
            del obs

            time.sleep(args.delay)

            state = get_state(env)

            total_reward += float(reward)

            steps += 1

            done = terminated or truncated

            if info.get("success", False):
                success = True

        print(
            f"episode={episode} "
            f"steps={steps} "
            f"total_reward={total_reward:.3f} "
            f"success={success}"
        )

    env.close()


if __name__ == "__main__":
    main()