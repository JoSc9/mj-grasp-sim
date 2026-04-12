from typing import List, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from tqdm import tqdm

from mgs.core.simualtion import MjSimulation
from mgs.gripper.base import MjShakableOpenCloseGripper
from mgs.obj.base import CollisionMeshObject
from mgs.util.geo.transforms import SE3Pose

XML = r"""
<mujoco>
    <compiler angle="radian" autolimits="true" />
    <option integrator="implicitfast" timestep="0.001"/>
    <compiler discardvisual="false"/>
    <option noslip_iterations="1"> </option>
    <option><flag multiccd="enable"/> </option>
    <option cone="elliptic" impratio="3" timestep="0.001" noslip_iterations="2" noslip_tolerance="1e-8" tolerance="1e-8"/>
    <option gravity="0 0 0" />
    {gripper}
    <worldbody>
        <light name="light:top" pos="0 0 0.3"/>
        <light name="light:right" pos="0.3 0 0"/>
        <light name="light:left" pos="-0.3 0 0"/>
        <body name="body:ground" pos="0.0 0 -1.0">
           <geom name="geom:ground" pos="0 0 0" rgba="1.0 1.0 1.0 0.0" size="1.0 1.0 0.02" type="box" density="500"/>
        </body>
    </worldbody>
    {object}
</mujoco>
"""


class GravitylessObjectGrasping(MjSimulation):
    def __init__(self, gripper: MjShakableOpenCloseGripper, obj: CollisionMeshObject, render: bool = False):
        self.gripper = gripper
        self.obj = obj
        self.gripper_xml, self.gripper_assets = gripper.to_xml()
        self.object_xml, self.object_assets = obj.to_xml()
        self.model_xml = XML.format(
            **{"gripper": self.gripper_xml, "object": self.object_xml}
        )

        self.model = mujoco.MjModel.from_xml_string(  # type: ignore
            self.model_xml, {**self.gripper_assets, **self.object_assets}
        )
        self.data = mujoco.MjData(self.model)  # type: ignore
        mujoco.mj_forward(self.model, self.data)  # type: 
        
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def idle_grasp(self, pose: SE3Pose, joints: np.ndarray):
        import mujoco.viewer

        # (Implementation remains the same)
        mujoco.mj_resetData(self.model, self.data)
        b2c = self.gripper.base_to_contact_transform()
        pose_processed = pose @ b2c
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )
        self.set_qpos(joints, gripper_joint_idxs)
        self.gripper.set_pose(self, pose_processed)

        mujoco.mj_forward(self.model, self.data)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while True:
                viewer.sync()
                mujoco.mj_step(self.model, self.data)

    def grasp_collision_mask(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
        with_padding: float | None = None,
    ) -> np.ndarray:
        if len(poses) != len(joints):
            raise ValueError(
                f"Number of poses ({len(poses)}) must match number of joint configurations ({len(joints)})."
            )
        if joints.shape[1] != len(self.gripper.get_actuator_joint_names()):
            raise ValueError(
                f"Joints array has incorrect dimension ({joints.shape[1]}), expected {len(self.gripper.get_actuator_joint_names())}."
            )

        collision_free_mask: List[bool] = []
        num_grasps = len(poses)
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )

        # prebuild the 7 local offsets we will apply BEFORE base-to-contact:
        # identity, and translations by ±padding along local x, y, z (no rotation)
        if with_padding is not None and with_padding > 0:
            zero = np.zeros(3, dtype=np.float32)
            qwxyz = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            )  # identity quat (wxyz)
            p = float(with_padding)
            deltas = [
                SE3Pose(zero, qwxyz, "wxyz"),  # original (no shift)
                SE3Pose(np.array([+p, 0.0, 0.0], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([-p, 0.0, 0.0], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([0.0, +p, 0.0], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([0.0, -p, 0.0], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([0.0, 0.0, +p], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([0.0, 0.0, -p], np.float32), qwxyz, "wxyz"),
            ]
        else:
            deltas = [None]  # sentinel meaning "no perturbation"

        initial_state = self.get_state()

        for i in range(num_grasps):
            all_clear = True

            for delta in deltas:
                # reset to a clean state before each check
                mujoco.mj_resetData(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)

                # compose: apply optional local translation BEFORE base-to-contact transform
                # world_T_grasp_perturbed = world_T_grasp @ T_local(delta)
                grasp_pose = poses[i] if delta is None else (poses[i] @ delta)

                # then move from gripper base to its contact frame
                pose_processed = grasp_pose @ self.gripper.base_to_contact_transform()

                # set joints and pose, then evaluate contacts
                self.set_qpos(joints[i], gripper_joint_idxs)
                self.gripper.set_pose(self, pose_processed)
                mujoco.mj_forward(self.model, self.data)

                if self.check_contact():
                    all_clear = False
                    break  # no need to test the remaining perturbations

            collision_free_mask.append(all_clear)
            self.set_state(initial_state)
        return np.array(collision_free_mask)

    def grasp_stability_evaluation_from_joints(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
        impulse_force=300.0,
        enough_stable=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Streamlined stability check:
          1) For each grasp: close gripper once and verify contact.
          2) Save that post-close state.
          3) Apply six one-step 25 N impulses in the *grasp frame* (±x, ±y, ±z).
             Each impulse: restore saved state -> apply force for 1 step -> clear -> check contact.
          4) Mark stable only if all six pass. No lift/shake here.

        Returns the boolean results array (pos/rot drift arrays omitted for speed; kept in signature for compatibility).
        """
        if len(poses) != len(joints):
            raise ValueError(
                f"Number of poses ({len(poses)}) must match number of joint configurations ({len(joints)})."
            )
        if joints.shape[1] != len(self.gripper.get_actuator_joint_names()):
            raise ValueError(
                f"Joints array has incorrect dimension ({joints.shape[1]}), expected {len(self.gripper.get_actuator_joint_names())}."
            )

        IMPULSE_FORCE_N = float(
            impulse_force
        )  # one-step force magnitude (N) -> impulse J = F*dt
        object_bid = self.model.body(self.obj.name).id  # apply at object COM

        results: List[bool] = []
        num_grasps = len(poses)
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )

        # keep the user's original sim state
        initial_state = self.get_state()
        try:
            for i in tqdm(range(num_grasps)):
                if enough_stable is not None and sum(results) >= enough_stable:
                    # early-out fill: mark remaining as not evaluated
                    results.append(False)
                    continue

                # --- close once and save the post-close state ---
                mujoco.mj_resetData(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)

                b2c = self.gripper.base_to_contact_transform()
                pose_processed = poses[i] @ b2c
                self.set_qpos(joints[i], gripper_joint_idxs)
                self.gripper.set_pose(self, pose_processed)
                mujoco.mj_forward(self.model, self.data)

                self.gripper.close_gripper_at(self, pose_processed)

                if self.viewer is not None:
                # Visualize closed state
                    self.viewer.sync()

                if not self.check_contact_with_object():
                    results.append(False)
                    continue

                # snapshot the post-close state for deterministic, repeatable kicks
                closed_state = self.get_state()

                # world rotation of the grasp frame
                Rg = pose_processed.to_mat()[:3, :3].astype(float)

                # local unit axes in grasp frame
                local_dirs = np.eye(3, dtype=float)
                dirs_world = np.concatenate(
                    [
                        Rg @ local_dirs[:, [0, 1, 2]],  # +x,+y,+z
                        -(Rg @ local_dirs[:, [0, 1, 2]]),
                    ],
                    axis=1,
                ).T  # -x,-y,-z
                # dirs_world: shape (6, 3)

                all_pass = True
                for d in dirs_world:
                    # restore saved state
                    self.set_state(closed_state)
                    mujoco.mj_forward(self.model, self.data)

                    F = IMPULSE_FORCE_N * d
                    for i in range(5):
                        self.data.xfrc_applied[object_bid, :3] += F
                        mujoco.mj_step(
                            self.model, self.data, nstep=1
                        )  # integrates one step
                        
                        if self.viewer is not None:
                            self.viewer.sync()
                        
                        self.data.xfrc_applied[object_bid, :] = (
                            0.0  # clear so it doesn't persist
                        )
                        mujoco.mj_step(
                            self.model, self.data, nstep=10
                        )  # integrates one step

                    mujoco.mj_step(self.model, self.data, nstep=500)

                    if self.viewer is not None:
                        self.viewer.sync()
                    
                    # check contact right after the kick
                    if not self.check_contact_with_object():
                        all_pass = False
                        break

                results.append(all_pass)

            results_arr = np.array(results, dtype=bool)
            return results_arr

        finally:
            # restore original sim state even if something throws
            self.set_state(initial_state)

    def get_object_transform(self, object_name: str):
        # (Implementation remains the same)
        jnt_adr_start = self.model.jnt("{}:joint".format(object_name)).qposadr[0].item()
        obj_position = np.copy(self.data.qpos[jnt_adr_start : jnt_adr_start + 3])
        obj_quat = np.copy(self.data.qpos[jnt_adr_start + 3 : jnt_adr_start + 7])
        return SE3Pose(
            obj_position.astype(np.float32), obj_quat.astype(np.float32), "wxyz"
        )

    def check_contact(self):
        return self.data.ncon != 0

    def check_contact_with_object(self):
        """
        As the geoms are ordered accordingly to the XML. We can simply
        check for contacts between obj geoms and gripper geoms by ids
        relative to the table (which is inbetween obj and gripper by construction)
        """
        table_id = self.model.geom("geom:ground").id
        for contact_pairs in self.data.contact.geom:
            if (contact_pairs[0] < table_id and contact_pairs[1] > table_id) or (
                contact_pairs[0] > table_id and contact_pairs[1] < table_id
            ):
                return True
        return False
