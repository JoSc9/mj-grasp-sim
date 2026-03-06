# Copyright (c) 2025 Robert Bosch GmbH
# Author: Roman Freiberg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from omegaconf import DictConfig

from mgs.gripper.allegro import GripperAllegro
from mgs.gripper.base import MjShakableOpenCloseGripper
from mgs.gripper.dexee import GripperDexee
from mgs.gripper.google import GripperGoogle
from mgs.gripper.panda import GripperPanda
from mgs.gripper.panda_gelsight_mini import GripperPandaGelSightMini
from mgs.gripper.robotiq2f85 import GripperRobotiq2f85
from mgs.gripper.shadow import GripperShadowRight
from mgs.gripper.static.allegro import StaticGripperAllegro
from mgs.gripper.static.dexee import StaticGripperDexee
from mgs.gripper.static.panda import StaticGripperPanda
from mgs.gripper.static.shadow import StaticGripperShadowRight
from mgs.gripper.static.vx300 import StaticGripperVX300
from mgs.gripper.vx300 import GripperVX300
from mgs.util.geo.transforms import SE3Pose


def get_gripper(cfg: DictConfig, default_pose=None) -> MjShakableOpenCloseGripper:
    pose = (
        SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz")
        if default_pose is None
        else default_pose
    )
    if cfg.name == "GoogleGripper":
        return GripperGoogle(pose)

    if cfg.name == "PandaGripper":
        return GripperPanda(pose)
    
    if cfg.name == "PandaGripperGelsightMini":
        return GripperPandaGelSightMini(pose)

    if cfg.name == "StaticPandaGripper":
        return StaticGripperPanda(pose)

    if cfg.name == "Robotiq2f85Gripper":
        return GripperRobotiq2f85(pose)

    if cfg.name == "ShadowHand":
        return GripperShadowRight(pose)

    if cfg.name == "StaticShadowHand":
        return StaticGripperShadowRight(pose)

    if cfg.name == "VXGripper":
        return GripperVX300(pose)

    if cfg.name == "StaticVXGripper":
        return StaticGripperVX300(pose)

    if cfg.name == "DexeeGripper":
        return GripperDexee(pose)

    if cfg.name == "StaticDexeeGripper":
        return StaticGripperDexee(pose)

    if cfg.name == "AllegroGripper":
        return GripperAllegro(pose)

    if cfg.name == "StaticAllegroGripper":
        return StaticGripperAllegro(pose)

    raise ValueError(f"Unknown gripper: {cfg.name}")
