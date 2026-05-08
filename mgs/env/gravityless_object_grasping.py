from typing import List, Tuple

import mujoco
import numpy as np
from tqdm import tqdm

# Visualization imports
try:
    import plotly.graph_objects as go
    import trimesh
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: plotly and/or trimesh not available. Visualization will be limited.")

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
    def __init__(self, gripper: MjShakableOpenCloseGripper, obj: CollisionMeshObject):
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
        mujoco.mj_forward(self.model, self.data)  # type: ignore

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
            
        self._visualize_grasp_results(poses, np.array([True] * len(poses)))  # visualize all poses before checking collisions

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
        
        # visualize all generated grasps
        results_placeholder = np.array([True] * num_grasps)  # placeholder for visualization
        self._visualize_grasp_results(poses, results_placeholder)

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
                        self.data.xfrc_applied[object_bid, :] = (
                            0.0  # clear so it doesn't persist
                        )
                        mujoco.mj_step(
                            self.model, self.data, nstep=10
                        )  # integrates one step

                    mujoco.mj_step(self.model, self.data, nstep=500)

                    # check contact right after the kick
                    if not self.check_contact_with_object():
                        all_pass = False
                        break

                results.append(all_pass)

            results_arr = np.array(results, dtype=bool)
            
            # visualize results on obj
            self._visualize_grasp_results(poses, results_arr)
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

    def _visualize_grasp_results(self, poses: List[SE3Pose], results: np.ndarray):
        """
        Visualize grasp poses as coordinate frames with object mesh using plotly.
        Green coordinate frames for successful grasps, red for failed grasps.
        """
        obj_pos = self.get_object_transform(self.obj.name).pos
        
        print(f"Visualizing {len(poses)} grasp poses:")
        print(f"  Successful grasps: {np.sum(results)} / {len(results)}")
        print(f"  Success rate: {np.sum(results) / len(results) * 100:.1f}%")
        
        if not VISUALIZATION_AVAILABLE:
            print("  Plotly/trimesh not available. Visualization skipped.")
            return
            
        try:
            # Create plotly figure
            fig = go.Figure()
            
            # Load and add object mesh
            self._add_object_mesh_to_plot(fig)
            
            # Add coordinate frames
            self._add_coordinate_frames_to_plot(fig, poses, results)
            
            # Configure layout
            fig.update_layout(
                title=f"Grasp Coordinate Frames - {np.sum(results)}/{len(results)} successful ({np.mean(results)*100:.1f}%)",
                scene=dict(
                    xaxis_title="X (m)",
                    yaxis_title="Y (m)", 
                    zaxis_title="Z (m)",
                    aspectmode="data",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5),
                        center=dict(x=0, y=0, z=0)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(itemsizing="constant")
            )
            
            # Show the plot
            fig.show()
            print(f"  Interactive 3D visualization opened in browser")
            
        except Exception as e:
            print(f"  Visualization error: {e}")
            print("  Falling back to basic console output")
            
    def _add_object_mesh_to_plot(self, fig):
        """Add object mesh to the plotly figure using trimesh."""
        try:
            # Load mesh from object file path
            if hasattr(self.obj, 'obj_file_path') and self.obj.obj_file_path:
                mesh = trimesh.load(self.obj.obj_file_path)
                
                # Get object transform
                obj_transform = self.get_object_transform(self.obj.name)
                transform_matrix = obj_transform.to_mat()
                
                # Transform mesh vertices
                vertices = mesh.vertices
                vertices_homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
                transformed_vertices = (transform_matrix @ vertices_homogeneous.T).T[:, :3]
                
                # Add mesh to plot
                fig.add_trace(go.Mesh3d(
                    x=transformed_vertices[:, 0],
                    y=transformed_vertices[:, 1],
                    z=transformed_vertices[:, 2],
                    i=mesh.faces[:, 0],
                    j=mesh.faces[:, 1],
                    k=mesh.faces[:, 2],
                    opacity=0.5,
                    color='lightblue',
                    name='Object Mesh',
                    showscale=False
                ))
                print(f"  Object mesh loaded: {len(vertices)} vertices, {len(mesh.faces)} faces")
            else:
                # Fallback: add a sphere at object center
                obj_pos = self.get_object_transform(self.obj.name).pos
                fig.add_trace(go.Scatter3d(
                    x=[obj_pos[0]],
                    y=[obj_pos[1]],
                    z=[obj_pos[2]],
                    mode='markers',
                    marker=dict(size=15, color='blue', opacity=0.7),
                    name='Object Center'
                ))
                print(f"  Object center marker added at {obj_pos}")
        except Exception as e:
            print(f"  Could not load object mesh: {e}")
            # Fallback to object center point
            obj_pos = self.get_object_transform(self.obj.name).pos
            fig.add_trace(go.Scatter3d(
                x=[obj_pos[0]],
                y=[obj_pos[1]],
                z=[obj_pos[2]],
                mode='markers',
                marker=dict(size=15, color='blue', opacity=0.7),
                name='Object Center'
            ))
            print(f"  Fallback: Object center marker added at {obj_pos}")
    
    def _add_coordinate_frames_to_plot(self, fig, poses: List[SE3Pose], results: np.ndarray):
        """Add coordinate frames for each grasp pose using SE3Pose .pos and .quat attributes."""
        frame_scale = 0.05  # Length of coordinate frame axes
        
        # Process successful and failed grasps separately
        for success_value, color, name in [(True, 'green', 'Successful Grasps'), 
                                          (False, 'red', 'Failed Grasps')]:
            
            matching_poses = [(pose, success) for pose, success in zip(poses, results) if success == success_value]
            
            if not matching_poses:
                continue
                
            # Prepare all line data for batch processing
            all_x, all_y, all_z = [], [], []
            centers = []
            
            for pose, _ in matching_poses:
                pos = pose.pos
                quat = pose.quat  # quaternion in wxyz format
                
                # Convert quaternion to rotation matrix
                # quat = [w, x, y, z] format
                w, x, y, z = quat[0], quat[1], quat[2], quat[3]
                
                # Quaternion to rotation matrix conversion
                rot_matrix = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                ])
                
                # Calculate coordinate frame axes
                x_axis_end = pos + frame_scale * rot_matrix[:, 0]  # X-axis (red)
                y_axis_end = pos + frame_scale * rot_matrix[:, 1]  # Y-axis (green)
                z_axis_end = pos + frame_scale * rot_matrix[:, 2]  # Z-axis (blue)
                
                # Add X-axis line
                all_x.extend([pos[0], x_axis_end[0], None])
                all_y.extend([pos[1], x_axis_end[1], None])
                all_z.extend([pos[2], x_axis_end[2], None])
                
                # Add Y-axis line
                all_x.extend([pos[0], y_axis_end[0], None])
                all_y.extend([pos[1], y_axis_end[1], None])
                all_z.extend([pos[2], y_axis_end[2], None])
                
                # Add Z-axis line
                all_x.extend([pos[0], z_axis_end[0], None])
                all_y.extend([pos[1], z_axis_end[1], None])
                all_z.extend([pos[2], z_axis_end[2], None])
                
                centers.append(pos)
            
            # Add all coordinate frame lines as a single trace
            if all_x:
                fig.add_trace(go.Scatter3d(
                    x=all_x,
                    y=all_y,
                    z=all_z,
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=f'{name} Frames',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # Add center points for reference
                centers = np.array(centers)
                fig.add_trace(go.Scatter3d(
                    x=centers[:, 0],
                    y=centers[:, 1], 
                    z=centers[:, 2],
                    mode='markers',
                    marker=dict(size=5, color=color, opacity=0.8,
                               line=dict(width=1, color='black')),
                    name=f'{name} Centers',
                    showlegend=False
                ))

