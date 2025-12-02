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

from omegaconf import DictConfig

from mgs.env.clutter_table import ClutterTableEnv


def get_env(cfg: DictConfig, gripper, obj_list, headless: bool=True):
    if cfg.name == "ClutterTable":
        env = ClutterTableEnv(gripper, objects=obj_list, headless=headless)
    else:
        raise ValueError(f"Unknown environment {cfg.name}")
    return env


def get_env_from_dict(cfg: DictConfig, scene_dict, headless: bool=True):
    if cfg.name == "ClutterTable":
        env = ClutterTableEnv.from_dict(scene_dict, headless=headless)
    else:
        raise ValueError(f"Unknown environment {cfg.name}")
    return env
