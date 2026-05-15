from __future__ import annotations

from dataclasses import dataclass

from minigrid.core.constants import COLORS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect


@dataclass(frozen=True)
class WarehouseProfile:
    size: int
    max_steps: int
    agent_pos: tuple[int, int]
    agent_dir: int
    pickup_pos: tuple[int, int]
    dropoff_pos: tuple[int, int]
    shelf_columns: tuple[int, ...]
    shelf_y_start: int
    shelf_y_end: int
    shelf_gap_y: int
    possible_blockages: tuple[tuple[int, int], ...]


WAREHOUSE_PROFILES: dict[str, WarehouseProfile] = {
    "tiny": WarehouseProfile(
        size=6,
        max_steps=80,
        agent_pos=(1, 1),
        agent_dir=1,
        pickup_pos=(1, 4),
        dropoff_pos=(4, 4),
        shelf_columns=(3,),
        shelf_y_start=2,
        shelf_y_end=5,
        shelf_gap_y=3,
        possible_blockages=((-1, -1), (1, 3), (2, 4), (3, 4)),
    ),
    "small": WarehouseProfile(
        size=8,
        max_steps=120,
        agent_pos=(1, 1),
        agent_dir=1,
        pickup_pos=(2, 6),
        dropoff_pos=(5, 6),
        shelf_columns=(3, 5),
        shelf_y_start=2,
        shelf_y_end=7,
        shelf_gap_y=4,
        possible_blockages=((-1, -1), (1, 4), (1, 5), (2, 5), (4, 6)),
    ),
    "medium": WarehouseProfile(
        size=10,
        max_steps=180,
        agent_pos=(1, 1),
        agent_dir=1,
        pickup_pos=(2, 8),
        dropoff_pos=(7, 8),
        shelf_columns=(3, 6),
        shelf_y_start=2,
        shelf_y_end=8,
        shelf_gap_y=5,
        possible_blockages=(
            (-1, -1),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 6),
            (4, 8),
            (5, 8),
            (6, 8),
        ),
    ),
    "large": WarehouseProfile(
        size=14,
        max_steps=300,
        agent_pos=(1, 1),
        agent_dir=1,
        pickup_pos=(2, 12),
        dropoff_pos=(11, 12),
        shelf_columns=(3, 6, 9),
        shelf_y_start=2,
        shelf_y_end=12,
        shelf_gap_y=7,
        possible_blockages=(
            (-1, -1),
            (1, 6),
            (1, 7),
            (1, 8),
            (2, 10),
            (4, 12),
            (5, 12),
            (7, 12),
            (8, 12),
            (10, 12),
        ),
    ),
}


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
    def __init__(self, size: str = "medium", render_mode: str | None = None):
        if size not in WAREHOUSE_PROFILES:
            valid_sizes = ", ".join(sorted(WAREHOUSE_PROFILES))
            raise ValueError(
                f"Unknown warehouse size '{size}'. Choose one of: {valid_sizes}"
            )

        self.size_name = size
        self.profile = WAREHOUSE_PROFILES[size]
        self.size = self.profile.size

        self.package_color = "blue"
        self.pickup_pos = self.profile.pickup_pos
        self.dropoff_pos = self.profile.dropoff_pos

        self.blocked_pos = (-1, -1)
        self.carrying_item = False

        mission_space = MissionSpace(
            mission_func=lambda: "pick up the package and deliver it to the drop-off station"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            max_steps=self.profile.max_steps,
            see_through_walls=True,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for shelf_x in self.profile.shelf_columns:
            for y in range(self.profile.shelf_y_start, self.profile.shelf_y_end):
                if y != self.profile.shelf_gap_y:
                    self.put_obj(Wall(), shelf_x, y)

        self.agent_pos = self.profile.agent_pos
        self.agent_dir = self.profile.agent_dir

        self.put_obj(Ball(self.package_color), *self.pickup_pos)
        self.put_obj(DropOff(self.package_color), *self.dropoff_pos)

        self.blocked_pos = self._rand_elem(list(self.profile.possible_blockages))

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
        self.mission = "pick up the package and deliver it to the drop-off station"

    def step(self, action):
        old_pos = tuple(self.agent_pos)

        obs, _, terminated, truncated, info = super().step(action)

        reward = -0.01
        new_pos = tuple(self.agent_pos)

        info["success"] = False
        info["picked_up"] = False
        info["blocked_pos"] = self.blocked_pos
        info["warehouse_size"] = self.size_name

        if action == self.actions.forward and new_pos == old_pos:
            reward -= 0.20

        if not self.carrying_item and new_pos == self.pickup_pos:
            self.carrying_item = True
            self.grid.set(*self.pickup_pos, None)
            reward += 1.0
            info["picked_up"] = True

        if self.carrying_item and new_pos == self.dropoff_pos:
            self.carrying_item = False
            reward += 5.0
            terminated = True
            info["success"] = True

        return obs, reward, terminated, truncated, info