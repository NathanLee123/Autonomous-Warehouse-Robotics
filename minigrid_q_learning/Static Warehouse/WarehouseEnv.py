from __future__ import annotations

from minigrid.core.actions import Actions
from minigrid.core.constants import COLORS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect


class DropOff(WorldObj):
    def __init__(self, color: str):
        super().__init__("floor", color)

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def render(self, img):
        fill_coords(
            img,
            point_in_rect(0.15, 0.85, 0.15, 0.85),
            COLORS[self.color],
        )


class DynamicBlockage(WorldObj):
    def __init__(self):
        super().__init__("wall", "red")

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def render(self, img):
        fill_coords(
            img,
            point_in_rect(0.10, 0.90, 0.10, 0.90),
            COLORS["red"],
        )


class WarehouseEnv(MiniGridEnv):
    def __init__(self, render_mode: str | None = None):
        self.size = 10

        self.package_color = "blue"

        self.pickup_pos = (2, 8)
        self.dropoff_pos = (7, 8)

        self.blocked_pos = (-1, -1)

        self.carrying_item = False

        # NEW: remember previous action
        self.previous_action = -1

        mission_space = MissionSpace(
            mission_func=lambda: "pick up the package and deliver it to the drop-off station"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            max_steps=250,
            see_through_walls=True,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Static warehouse shelves / racks
        for y in range(2, 8):
            if y != 5:
                self.put_obj(Wall(), 3, y)
                self.put_obj(Wall(), 6, y)

        # Fixed agent start
        self.agent_pos = (1, 1)
        self.agent_dir = 1

        # Fixed package and drop-off
        self.put_obj(Ball(self.package_color), *self.pickup_pos)
        self.put_obj(DropOff(self.package_color), *self.dropoff_pos)

        # Dynamic obstacle locations
        possible_blockages = [
            (-1, -1),

            # Blocks route from start -> package
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 6),

            # Blocks route from package -> dropoff
            (4, 8),
            (5, 8),
            (6, 8),
        ]

        self.blocked_pos = self._rand_elem(possible_blockages)

        if self.blocked_pos != (-1, -1):
            if (
                self.grid.get(*self.blocked_pos) is None
                and self.blocked_pos != self.agent_pos
                and self.blocked_pos != self.pickup_pos
                and self.blocked_pos != self.dropoff_pos
            ):
                self.put_obj(DynamicBlockage(), *self.blocked_pos)
            else:
                self.blocked_pos = (-1, -1)

        self.carrying_item = False
        self.previous_action = -1

        self.mission = "pick up the package and deliver it to the drop-off station"

    def step(self, action):
        old_pos = tuple(self.agent_pos)

        # Successful delivery
        if action == Actions.drop and self.carrying is not None:
            carried_color = self.carrying.color

            if (
                self.carrying_item
                and carried_color == self.package_color
                and old_pos == self.dropoff_pos
            ):
                self.carrying = None
                self.carrying_item = False

                obs = self.gen_obs()

                reward = 5.0
                terminated = True
                truncated = False

                info = {
                    "success": True,
                    "delivered_color": carried_color,
                    "blocked_pos": self.blocked_pos,
                }

                self.previous_action = int(action)

                return obs, reward, terminated, truncated, info

        obs, original_reward, terminated, truncated, info = super().step(action)

        reward = -0.01

        info["success"] = False
        info["blocked_pos"] = self.blocked_pos

        new_pos = tuple(self.agent_pos)

        # Stronger turn penalty
        if action in (Actions.left, Actions.right):
            reward -= 0.05

        # Stronger blocked-forward penalty
        if action == Actions.forward and new_pos == old_pos:
            reward -= 0.20

        # Pickup reward
        if action == Actions.pickup:
            if self.carrying is not None and not self.carrying_item:
                self.carrying_item = True
                reward += 0.10
            elif self.carrying is None:
                reward -= 0.10

        # Invalid or wrong drop
        if action == Actions.drop:
            if self.carrying is None:
                reward -= 0.10
            elif old_pos != self.dropoff_pos:
                reward -= 0.50

        # NEW: save previous action
        self.previous_action = int(action)

        return obs, reward, terminated, truncated, info