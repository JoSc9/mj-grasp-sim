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

import os
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np

from mgs.core.simualtion import MjSimulation
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose

from .base import MjScannable, MjShakableOpenCloseGripper

# The XML template string is derived from MuJoCo Menagerie
# (https://github.com/google-deepmind/mujoco_menagerie/tree/469893211c41d5da9c314f5ab58059fa17c8e360)
# Copyright 2024 Franka Robotics License: Apache-2.0
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
XML = """
  <default>
    <default class="panda">
      <material specular="0.5" shininess="0.25"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>

      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
        <default class="fingertip_pad_collision_1">
          <geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445" friction="1.5 0.3 0.1"/>
        </default>
        <default class="fingertip_pad_collision_2">
          <geom type="box" size="0.003 0.002 0.003" pos="0.0055 0.002 0.05" friction="1.5 0.3 0.1"/>
        </default>
        <default class="fingertip_pad_collision_3">
          <geom type="box" size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05" friction="1.5 0.3 0.1"/>
        </default>
        <default class="fingertip_pad_collision_4">
          <geom type="box" size="0.003 0.002 0.0035" pos="0.0055 0.002 0.0395" friction="1.5 0.3 0.1"/>
        </default>
        <default class="fingertip_pad_collision_5">
          <geom type="box" size="0.003 0.002 0.0035" pos="-0.0055 0.002 0.0395" friction="1.5 0.3 0.1"/>
        </default>
      </default>
    </default>
  </default>

  <asset>
    <material class="panda" name="white" rgba="1 1 1 1"/>
    <material class="panda" name="off_white" rgba="0.901961 0.921569 0.929412 1"/>
    <material class="panda" name="black" rgba="0.25 0.25 0.25 1"/>

    <!-- Collision meshes -->
    <mesh name="hand_c" file="hand.stl"/>

    <!-- Visual meshes -->
    <mesh file="hand_0.obj"/>
    <mesh file="hand_1.obj"/>
    <mesh file="hand_2.obj"/>
    <mesh file="hand_3.obj"/>
    <mesh file="hand_4.obj"/>
    <mesh file="finger_0.obj"/>
    <mesh file="finger_1.obj"/>

    <!-- gelsight_mini sensor -->
    <mesh name="gelsight_shell" file="gsmini_shell.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
    <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
    <body name="hand" childclass="panda" quat="{quaternion}" pos="{position}">
      <freejoint name="freejoint"/>
      <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
      <geom mesh="hand_0" material="off_white" class="visual"/>
      <geom mesh="hand_1" material="black" class="visual"/>
      <geom mesh="hand_2" material="black" class="visual"/>
      <geom mesh="hand_3" material="white" class="visual"/>
      <geom mesh="hand_4" material="off_white" class="visual"/>
      <geom mesh="hand_c" class="collision"/>
      <body name="left_finger" pos="0 0 0.0584">
        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
        <joint name="finger_joint1" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="0.0 0.04" damping="100" armature="1.0" frictionloss="1.0"/>
        <geom mesh="finger_0" material="off_white" class="visual"/>
        <geom mesh="finger_1" material="black" class="visual"/>
        <geom mesh="finger_0" class="collision" name="panda_col_1" />
        <geom type="mesh" mesh="gelsight_shell" material="black" pos="0 0.005 0.04" quat="0 0.707 0 0.707" condim="6" rgba="1 1 1 0.3" mass="0.03" solimp="0.09 0.95 0.001 0.5 2" solref="1 1" name="col_gelsight_left" group="4"/>
        <geom type="box" size="0.01 0.001 0.01" pos="0 -0.01325 0.04" priority="1" rgba="0 0 1 0" contype="1" conaffinity="1" name="hard_stop_left" solimp="0.99 0.99 0.001" solref="0.001 1" condim="3"/>

        <camera name="tactile_cam_left" pos="0 0 0.04" quat="0.707 -0.707 0 0" fovy="35" resolution="640 480"/>
      </body>
      <body name="right_finger" pos="0 -0.04 0.0584" quat="0 0 0 1">
        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
        <joint name="finger_joint2" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="-0.04 0.0" damping="100" armature="1.0" frictionloss="1.0"/>
        <geom mesh="finger_0" material="off_white" class="visual"/>
        <geom mesh="finger_1" material="black" class="visual"/>
        <geom mesh="finger_0" class="collision" name="panda_col_7" />
        <geom type="mesh" mesh="gelsight_shell" material="black" pos="0 0.005 0.04" quat="0 0.707 0 0.707" condim="6" mass="0.03" solimp="0.09 0.95 0.001 0.5 2" solref="1 1" rgba="1 0 0 0.5" name="col_gelsight_right" group="4"/>
        <geom type="box" size="0.01 0.001 0.01" pos="0 -0.01324 0.04" priority="1" rgba="0 1 0 0" contype="1" conaffinity="1" name="hard_stop_right" solimp="0.99 0.99 0.001" solref="0.001 1" condim="3"/>

        <camera name="tactile_cam_right" pos="0 0 0.04" quat="0.707 -0.707 0 0" fovy="35" resolution="640 480"/>

      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="hand" body2="left_finger"/>
    <exclude body1="hand" body2="right_finger"/>
  </contact>

  <equality>
    <weld body1="mocap" body2="hand"/>
  </equality>

  <actuator>
    <position ctrllimited="true" ctrlrange="0.0 0.04" joint="finger_joint1" kp="1000" name="gripper_finger_joint1" forcelimited="true" forcerange="-5 5"/>
    <position ctrllimited="true" ctrlrange="-0.04 0.0" joint="finger_joint2" kp="1000" name="gripper_finger_joint2" forcelimited="true" forcerange="-5 5"/>
  </actuator>
"""


class GripperPandaGelSightMini(MjShakableOpenCloseGripper, MjScannable):
    MIN_WIDTH_TARGET = 0.0  # Target closed width before clamping
    MAX_WIDTH = 0.08  # Max open width (8cm)
    # Elastic part of gelsight mini sensor before hard stop (-> value from: https://gitlab.sdu.dk/pengu20/mj_sim/-/blob/f9336f6d4b8d44384ce7ebab3739813f66d34dfc/sensors/gelsight_mini/gelsight_mini.py)
    ELASTOMER_THICKNESS = 0.004 
    # Thickness of gelsight mini sensor (computed by tactile_sensing/sensor_thickness.py)
    SENSOR_THICKNESS = 0.01824
    # Minimum physical width clamp (Hard component of Sensor = SENSOR_THICKNESS - ELASTOMER_THICKNESS)
    MIN_WIDTH_CLAMP = 2*(SENSOR_THICKNESS-ELASTOMER_THICKNESS)

    # Joint limits from XML
    Q1_RANGE = [0.0, 0.04]
    # TODO: Anpassen nur bis max schließen
    Q2_RANGE = [-0.04, 0.0]

    def __init__(self, pose: SE3Pose):
        super().__init__(pose, "hand")  # "hand" is the base body name in the Panda XML

    def to_xml(self) -> Tuple[str, Dict[str, Any]]:
        """
        Generates the MuJoCo XML snippet and assets for the Panda gripper.
        Note: This assumes the global XML variable is defined containing the template.
        """
        # This part remains the same, it just formats the XML string
        pos = "{} {} {}".format(self.pos[0], self.pos[1], self.pos[2])
        quat = "{} {} {} {}".format(
            self.quat[0], self.quat[1], self.quat[2], self.quat[3]
        )
        # We need the XML string from the outer scope or defined here
        # For now, assume it's accessible as `XML`
        try:
            formatted_xml = XML.format(position=pos, quaternion=quat)
        except NameError:
            raise RuntimeError(
                "Panda XML template string 'XML' is not defined in the scope of GripperPanda.to_xml"
            )

        ASSETS = dict()
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
        abs_asset_path = os.path.join(repo_root, "asset")

        base_path = os.path.join(abs_asset_path, "panda")
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Panda asset directory not found at {base_path}")
        for file_name in os.listdir(base_path):
            path = os.path.join(base_path, file_name)
            if os.path.isfile(path):  # Ensure it's a file
                try:
                    with open(path, "rb") as f:
                        ASSETS[file_name] = f.read()
                except IOError as e:
                    print(f"Warning: Could not read asset file {path}: {e}")

        # Add GelSight Assets
        sensor_path = os.path.join(abs_asset_path, "sensors", "gelsight_mini", "gsmini_shell.stl")
        if os.path.isfile(sensor_path):
          try:
            with open(sensor_path, "rb") as f:
              ASSETS["gsmini_shell.stl"] = f.read()
          except IOError as e:
            print(f"Warning: Could not read sensor file {sensor_path}: {e}")

        return (formatted_xml, ASSETS)

    def base_to_contact_transform(self) -> SE3Pose:
        pos = np.array([0, 0, -0.09815])
        quat = np.array([0.707106781, 0.0, 0.0, 0.707106781])
        return SE3Pose(pos, quat, type="wxyz")

    def open_gripper(self, sim: MjSimulation):
        """Opens the gripper to its maximum width (8cm) based on original command."""
        # Use the command known to open the gripper fully
        target_q1 = self.Q1_RANGE[1]  # 0.04
        target_q2 = self.Q2_RANGE[0]  # 0.0
        qpos_indices = sim.get_joint_idxs(self.get_actuator_joint_names())

        # Set qpos and ctrl
        sim.data.qpos[qpos_indices[0]] = target_q1
        sim.data.qpos[qpos_indices[1]] = target_q2
        sim.data.ctrl[0] = target_q1
        sim.data.ctrl[1] = target_q2
        mujoco.mj_forward(sim.model, sim.data)

    def close_gripper(self, sim: MjSimulation):
        """Commands the gripper to its fully closed state based on original command."""
        # Use dynamic width computation with negative distance to ensure to return MIN_WIDTH_CLAMP
        # If width_to_joints is called with width=0.00 -> Gripper only closes to 2*SENSOR_THICKNESS -> Object would only be touched and might slip
        target_joints = self.width_to_joints(-1.0)
        
        sim.data.ctrl[0] = target_joints[0]
        sim.data.ctrl[1] = target_joints[1]
        # # Use the command known to close the gripper fully
        # target_q1 = self.Q1_RANGE[0]  # 0.0
        # target_q2 = self.Q2_RANGE[0]  # -0.04
        # sim.data.ctrl[0] = target_q1
        # sim.data.ctrl[1] = target_q2

    def width_to_joints(self, width: float):
        SENSOR_THICKNESS = 0.01824
        adjusted_width = width + 2 * SENSOR_THICKNESS
        clamped_width = np.clip(adjusted_width, self.MIN_WIDTH_CLAMP, self.MAX_WIDTH)
        target_q1 = clamped_width / 2.0
        target_q2 = -0.04 + (clamped_width / 2.0)
        target_q1 = np.clip(target_q1, self.Q1_RANGE[0], self.Q1_RANGE[1])
        target_q2 = np.clip(target_q2, self.Q2_RANGE[0], self.Q2_RANGE[1])
        return np.stack([target_q1, target_q2], axis=-1)

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        """
        Sets the gripper base pose and commands the fingers to close fully.
        Uses the original known control signals for closing.
        """
        # Set the base pose using mocap
        # Use [:] to modify in place if sim.data.mocap_pos is a view
        sim.data.mocap_pos[:] = np.copy(pose.pos)
        sim.data.mocap_quat[:] = np.copy(pose.quat)

        # # Command the fingers to close using the original known control values
        # close_ctrl_signal = np.array([0.0, -0.04])
        # sim.data.ctrl[:] = close_ctrl_signal  # Set ctrl for both actuators
        
        # Close gripper to negative distance to ensure to return MIN_WIDTH_CLAMP
        close_ctrl_signal = self.width_to_joints(-1.0)
        sim.data.ctrl[:] = close_ctrl_signal

        # Step the simulation to allow the controller to close the fingers
        # Keep existing step count
        mujoco.mj_step(sim.model, sim.data, nstep=1000)
    
    def set_finger_max_force(self, sim: MjSimulation, max_force: float):
        """Sets the maximum force for the gripper finger actuators."""
        sim.model.actuator_forcerange[:] = np.array([-max_force, max_force])

    def get_actuator_joint_names(self) -> List[str]:
        """Returns the names of the joints directly controlled by actuators."""
        return ["finger_joint1", "finger_joint2"]

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        """Gets the qpos indices for the 6-DOF free joint of the base ('hand')."""
        # Ensure the freejoint name matches the one in the XML ('freejoint')
        try:
            start_idx = sim.get_joint_idxs(["freejoint"])[0]
        except IndexError:
            raise RuntimeError(
                "Could not find 'freejoint' qpos index for the Panda base."
            )
        # Free joint has 7 qpos values (3 pos, 4 quat)
        return list(range(start_idx, start_idx + 7))

    def get_render_options(self):
        """Returns default visualization options."""
        options = mujoco.MjvOption()
        return options

    def _clamp_width(self, width: np.array) -> float:
        """Clamps the desired width to the gripper's operational range."""
        
        SENSOR_THICKNESS = 0.01824
        
        # Add sensor thickness of both sensors to the desired width
        adjusted_width = width + (2 * SENSOR_THICKNESS)

        # TODO: Check if an additional buffer is required -> open gripper a little bit more than necessary
        # Padding of 0.008 m 
        return np.clip(adjusted_width+0.008, self.MIN_WIDTH_CLAMP, self.MAX_WIDTH)