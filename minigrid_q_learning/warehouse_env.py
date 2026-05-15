#!/usr/bin/env python3
"""Custom MiniGrid warehouse environments for reinforcement learning experiments.

This module defines a family of warehouse-style grid environments where an
agent must:
    - navigate around shelf obstacles
    - pick up a package
    - deliver it to a goal/drop-off location

Features:
    - procedurally generated warehouse aisle layouts
    - custom pickup package object with modified rendering
    - reward shaping for navigation progress and successful delivery
    - multiple environment sizes ranging from tiny to large
    - compatibility with MiniGrid and Gymnasium APIs

Included environments:
    - WarehouseGridTiny
    - WarehouseGridSmall
    - WarehouseGridMedium
    - WarehouseGridLarge

These environments are designed for experimenting with tabular Q-learning
and other reinforcement learning algorithms in structured navigation tasks.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Ball, Goal, Wall
from minigrid.utils.rendering import fill_coords, point_in_circle


class Package(Ball):
    """Package object that can be picked up by the agent."""

    def __init__(self, color: str = "purple"):
        super().__init__(color)

    def render(self, img):
        # Use a pink-looking fill even though the internal color uses a supported code.
        pink = (255, 102, 204)
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), pink)


class WarehouseGridEnv(MiniGridEnv):
    """Warehouse grid environment with shelves (walls) arranged in aisles."""

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        aisle_width: int = 2,
        shelf_height: int = 4,
        render_mode: str | None = None,
    ):
        """Initialize the warehouse environment.

        Args:
            width: Grid width
            height: Grid height
            aisle_width: Space between shelves (where agent can walk)
            shelf_height: Height of each shelf/obstacle block
            render_mode: Rendering mode (None, "human", "rgb_array")
        """
        self.aisle_width = aisle_width
        self.shelf_height = shelf_height

        mission_space = MissionSpace(
            mission_func=lambda: "pick up the package and take it to the drop-off location"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=width,
            max_steps=6 * width * height,
            render_mode=render_mode,
            see_through_walls=(render_mode == "human"),
        )

        self.width = width
        self.height = height

    def _gen_grid(self, width: int, height: int) -> None:
        """Generate the warehouse grid with shelves, aisles, and dock areas."""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Create horizontal shelves with repeating aisles and cross-aisles.
        shelf_rows = list(range(2, height - 2, 2))
        for idx, row_y in enumerate(shelf_rows):
            # leave a gap in each shelf row for cross-aisle access
            gap_x = 2 + (idx % (width - 4))
            for x in range(2, width - 2):
                if x == gap_x:
                    continue
                self.grid.set(x, row_y, Wall())

        def sample_empty_position(exclude: set[tuple[int, int]] | None = None, min_dist: int = 0) -> tuple[int, int]:
            exclude = exclude or set()
            candidates = []
            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    if self.grid.get(x, y) is None and (x, y) not in exclude:
                        candidates.append((x, y))
            self.np_random.shuffle(candidates)
            for pos in candidates:
                if min_dist == 0:
                    return pos
                if all(abs(pos[0] - ox) + abs(pos[1] - oy) > min_dist for ox, oy in exclude):
                    return pos
            if min_dist > 0:
                return sample_empty_position(exclude=exclude, min_dist=0)
            raise ValueError("No valid empty position found")

        self.agent_pos = np.array(sample_empty_position())
        self.agent_dir = self.np_random.integers(0, 4)

        self.package_pos = sample_empty_position(exclude={tuple(self.agent_pos)})
        self.grid.set(*self.package_pos, Package())

        self.goal_pos = sample_empty_position(exclude={tuple(self.agent_pos), self.package_pos}, min_dist=4)
        self.grid.set(*self.goal_pos, Goal())

    def step(self, action: int):
        prev_carrying = self.carrying
        prev_agent_pos = tuple(self.agent_pos)
        prev_target = self.goal_pos if prev_carrying is not None else self.package_pos
        prev_target_dist = abs(prev_target[0] - prev_agent_pos[0]) + abs(prev_target[1] - prev_agent_pos[1])

        obs, reward, terminated, truncated, info = super().step(action)

        agent_pos = tuple(self.agent_pos)

        if action == self.actions.pickup and self.carrying is not None and prev_carrying is None:
            reward += 1.0
            info["package_picked"] = True

        if agent_pos == self.goal_pos:
            if self.carrying is not None:
                terminated = True
                reward += 5.0
                info["success"] = True
            else:
                terminated = False
                reward = 0.0

        if not terminated and not (action == self.actions.pickup and self.carrying is not None and prev_carrying is None):
            current_target = self.goal_pos if self.carrying is not None else self.package_pos
            current_dist = abs(current_target[0] - agent_pos[0]) + abs(current_target[1] - agent_pos[1])
            reward += 0.05 * (prev_target_dist - current_dist)
            reward -= 0.01

        return obs, reward, terminated, truncated, info


class WarehouseGridSmall(WarehouseGridEnv):
    """Small warehouse (8x8) with wider aisles for easier learning."""

    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=8,
            height=8,
            aisle_width=3,  # Wider aisles
            shelf_height=2,
            render_mode=render_mode,
        )


class WarehouseGridTiny(WarehouseGridEnv):
    """Tiny warehouse (6x6) for quick testing and initial learning."""

    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=6,
            height=6,
            aisle_width=3,  # Very wide aisles
            shelf_height=1,  # Minimal obstacles
            render_mode=render_mode,
        )


class WarehouseGridMedium(WarehouseGridEnv):
    """Medium warehouse (10x10) with balanced difficulty."""

    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=10,
            height=10,
            aisle_width=2,
            shelf_height=2,
            render_mode=render_mode,
        )


class WarehouseGridLarge(WarehouseGridEnv):
    """Large warehouse (14x14) with narrow aisles for challenging learning."""

    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=14,
            height=14,
            aisle_width=2,
            shelf_height=3,
            render_mode=render_mode,
        )


if __name__ == "__main__":
    # Test the environment
    env = WarehouseGridSmall()
    obs, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
