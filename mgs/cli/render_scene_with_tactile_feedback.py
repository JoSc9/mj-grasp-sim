import os
import cv2  # Added for visualization
import mujoco.viewer  # Added for live MuJoCo rendering
from copy import deepcopy

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig

from mgs.env.selector import get_env_from_dict
from mgs.sensors.gelsight_mini.gelsight_mini import GelSightMini
from mgs.util.geo.transforms import SE3Pose


class DummyArgs:
    save_dir = "./"
    cam_width = 640
    cam_height = 480


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_grasp_files(hash_path: str):
    grasp_files = []
    for fname in sorted(os.listdir(hash_path)):
        if not fname.endswith(".npz"):
            continue
        if fname == "scene.npz":
            continue
        if "_collision" in fname:
            continue

        fpath = os.path.join(hash_path, fname)
        label = "failed" if "_failed" in fname else "stable"
        grasp_files.append((fpath, label))

    return grasp_files

def _derive_obj_name(fname: str) -> str:
    stem = fname.replace(".npz", "")
    for suffix in ("_failed", "_collision"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="render_scene_with_tactile_feedback")
def main(cfg: DictConfig):
    # -----------------------------------------------------------------------
    # 0. Feature Flags
    # -----------------------------------------------------------------------
    SAVE_TACTILE = bool(getattr(cfg, "save_tactile", True))
    VISUALIZE = bool(getattr(cfg, "visualize", False))

    print(f"[INFO] Configuration Settings:")
    print(f"       -> Save Tactile Data: {SAVE_TACTILE}")
    print(f"       -> Live Visualization: {VISUALIZE}")

    if VISUALIZE:
        print("\n[INFO] Controls during visualization:")
        print("       [ q ] - Quit entirely")
        print("       [ n ] - Skip entire scene")
        print("       [ l ] - Skip to next label group")
        print("       [ s ] - Skip to next individual grasp")

    # -----------------------------------------------------------------------
    # 1. Path setup
    # -----------------------------------------------------------------------
    output_dir = os.getenv("MGS_OUTPUT_DIR")
    if not output_dir:
        output_dir = os.path.expanduser("~/mj_data/out")
        
    base_dir = os.path.join(output_dir, "02_scene")
    gripper_dir = os.path.join(base_dir, cfg.gripper.name)

    if not os.path.exists(gripper_dir):
        print(f"[Error] Gripper directory not found: {gripper_dir}")
        return

    hash_dirs = [
        d for d in sorted(os.listdir(gripper_dir))
        if os.path.isdir(os.path.join(gripper_dir, d))
    ]
    if not hash_dirs:
        print(f"[Warning] No hash folders found in {gripper_dir}")
        return

    dummy_args = DummyArgs()
    num_scenes = len(hash_dirs)

    STEPS_CLOSE = int(getattr(cfg, "steps_close", 150))
    STEPS_LIFT = int(getattr(cfg, "steps_lift", 500))
    LIFT_DIST = float(getattr(cfg, "lift_dist", 0.3))
    RENDER_INTERVAL = int(getattr(cfg, "render_interval", 1))

    global_quit = False

    # -----------------------------------------------------------------------
    # 2. Iterate over scene hash directories
    # -----------------------------------------------------------------------
    for scene_count, hash_dir in enumerate(hash_dirs, 1):
        if global_quit:
            break

        hash_path = os.path.join(gripper_dir, hash_dir)
        scene_path = os.path.join(hash_path, "scene.npz")

        if not os.path.exists(scene_path):
            continue
        
        
        print(f"\n{'='*60}")
        print(f"Processing Scene {scene_count}/{num_scenes} | Hash: {hash_dir}")
        print(f"{'='*60}")

        scene = np.load(scene_path, allow_pickle=True)
        scene_dict = scene["scene_definition"].item()

        env = get_env_from_dict(cfg.env, deepcopy(scene_dict))
        model = env.model
        mj_data = env.data
        gripper = env.gripper

        model.vis.map.znear = 0.00001
        model.vis.map.zfar = 50.0

        settled_state = env.get_state()

        left_sensor = GelSightMini(args=dummy_args, model=model, data=mj_data, cam_name="tactile_cam_left")
        right_sensor = GelSightMini(args=dummy_args, model=model, data=mj_data, cam_name="tactile_cam_right")

        grasp_files = _load_grasp_files(hash_path)
        if not grasp_files:
            continue

        # Initialize viewer for this specific scene
        viewer = None
        if VISUALIZE:
            viewer = mujoco.viewer.launch_passive(model, mj_data)

        scene_skip = False
        current_label = None

        # -------------------------------------------------------------------
        # 6. Iterate over per-object grasp files
        # -------------------------------------------------------------------
        for grasp_fpath, grasp_label in grasp_files:
            if global_quit:
                break

            # If the user chose to skip to the next label type, bypass until the label changes
            if current_label is not None and grasp_label == current_label:
                continue
            else:
                current_label = None # Reset flag once a new label is hit

            grasp_fname = os.path.basename(grasp_fpath)
            print(f"\n  >> Grasp file: {grasp_fname}  (label={grasp_label})")

            try:
                grasp_data = np.load(grasp_fpath)
                poses_obj = grasp_data["pose"]
                joints_all = grasp_data["joints"]
            except Exception as exc:
                continue

            if poses_obj.ndim == 2:
                poses_obj = poses_obj[None]
                joints_all = joints_all[None]

            num_grasps = len(poses_obj)
            stem = _derive_obj_name(grasp_fname)
            
            obj_name = next((name for name in env.object_names if name in stem or stem in name), env.object_names[0])

            if SAVE_TACTILE:
                tactile_out_dir = os.path.join(hash_path, "tactile_feedback", stem)
                os.makedirs(tactile_out_dir, exist_ok=True)

            label_skip = False

            # ---------------------------------------------------------------
            # 7. Iterate over individual grasps
            # ---------------------------------------------------------------
            for grasp_idx in range(num_grasps):
                if global_quit or scene_skip or label_skip:
                    break

                pose_mat = poses_obj[grasp_idx]
                joint_tgt = joints_all[grasp_idx]

                # --- TARGET FILE PATH CHECK ---
                grasp_out_path = os.path.join(tactile_out_dir, f"grasp_{grasp_idx:05d}.npz") if SAVE_TACTILE else None
                lift_out_path = os.path.join(tactile_out_dir, f"lift_{grasp_idx:05d}.npz") if SAVE_TACTILE else None
                
                # Check if files already exist
                files_exist = (grasp_out_path and os.path.exists(grasp_out_path)) and (lift_out_path and os.path.exists(lift_out_path))
                
                if files_exist:
                    print(f"    Grasp {grasp_idx + 1}/{num_grasps} (label={grasp_label}) [ALREADY SAVED - VISUALIZING ONLY]")
                    skip_saving_this_grasp = True
                else:
                    print(f"    Grasp {grasp_idx + 1}/{num_grasps} (label={grasp_label})")
                    skip_saving_this_grasp = False


                env.set_state(settled_state)
                mujoco.mj_forward(model, mj_data)

                # 1. Parse the saved pose (It is already a world pose!)
                world_se3 = SE3Pose.from_mat(pose_mat[None], type="wxyz")[0]

                # 2. Apply the gripper base-to-contact offset
                b2c = gripper.base_to_contact_transform()
                gripper_base_pose = world_se3 @ b2c

                # 3. Set the pose
                gripper_joint_idxs = env.get_joint_idxs(gripper.get_actuator_joint_names())
                env.set_qpos(joint_tgt, gripper_joint_idxs)
                gripper.set_pose(env, gripper_base_pose)
                mujoco.mj_forward(model, mj_data)


                close_ctrl = gripper.width_to_joints(-1.0)
                mj_data.ctrl[:] = close_ctrl

                grasp_left_imgs, grasp_right_imgs = [], []
                contact_step = -1
                contact_frame_idx = -1
                skip_grasp = False

                # PHASE 1: Closing
                for step in range(STEPS_CLOSE):
                    mujoco.mj_step(model, mj_data)

                    if VISUALIZE and viewer.is_running():
                        viewer.sync()

                    if contact_step == -1 and env.check_gripper_contact():
                        contact_step = step
                        contact_frame_idx = step // RENDER_INTERVAL

                    if step % RENDER_INTERVAL == 0 and (SAVE_TACTILE or VISUALIZE):
                        img_l = left_sensor.tactile_image
                        img_r = right_sensor.tactile_image
                        
                        if img_l is not None and img_r is not None:
                            if SAVE_TACTILE and not skip_saving_this_grasp:
                                grasp_left_imgs.append(np.copy(img_l))
                                grasp_right_imgs.append(np.copy(img_r))
                            
                            if VISUALIZE:
                                cv2.imshow("Left Tactile", cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))
                                cv2.imshow("Right Tactile", cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):
                                    global_quit = True
                                    break
                                elif key == ord('s'):
                                    skip_grasp = True
                                    break
                                elif key == ord('l'):
                                    current_label = grasp_label
                                    label_skip = True
                                    break
                                elif key == ord('n'):
                                    scene_skip = True
                                    break

                if global_quit or skip_grasp or scene_skip or label_skip:
                    continue

                if SAVE_TACTILE and not skip_saving_this_grasp:
                    fully_closed_frame_idx = max(0, len(grasp_left_imgs) - 1)
                    grasp_out_path = os.path.join(tactile_out_dir, f"grasp_{grasp_idx:05d}.npz")
                    np.savez_compressed(
                        grasp_out_path,
                        left_cam=np.array(grasp_left_imgs, dtype=np.uint8),
                        right_cam=np.array(grasp_right_imgs, dtype=np.uint8),
                        contact_frame_idx=contact_frame_idx,
                        contact_step=contact_step,
                        fully_closed_frame_idx=fully_closed_frame_idx,
                        grasp_label=grasp_label,
                        pose_world=world_se3.to_mat(),
                        joints=joint_tgt,
                    )

                # PHASE 2: Lifting
                start_pos_lift = np.copy(mj_data.mocap_pos[0, :])
                lift_target_z = start_pos_lift[2] + LIFT_DIST
                lift_left_imgs, lift_right_imgs = [], []
                contact_lost_step = -1

                for step in range(STEPS_LIFT):
                    mj_data.ctrl[:] = close_ctrl
                    alpha = step / STEPS_LIFT
                    mj_data.mocap_pos[0, 2] = start_pos_lift[2] + (lift_target_z - start_pos_lift[2]) * alpha
                    mujoco.mj_step(model, mj_data)

                    if VISUALIZE and viewer.is_running():
                        viewer.sync()

                    if contact_lost_step == -1 and not env.check_gripper_contact():
                        contact_lost_step = step

                    if step % RENDER_INTERVAL == 0 and (SAVE_TACTILE or VISUALIZE):
                        img_l = left_sensor.tactile_image
                        img_r = right_sensor.tactile_image
                        
                        if img_l is not None and img_r is not None:
                            if SAVE_TACTILE and not skip_saving_this_grasp:
                                lift_left_imgs.append(np.copy(img_l))
                                lift_right_imgs.append(np.copy(img_r))
                            
                            if VISUALIZE:
                                cv2.imshow("Left Tactile", cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))
                                cv2.imshow("Right Tactile", cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):
                                    global_quit = True
                                    break
                                elif key == ord('s'):
                                    skip_grasp = True
                                    break
                                elif key == ord('l'):
                                    current_label = grasp_label
                                    label_skip = True
                                    break
                                elif key == ord('n'):
                                    scene_skip = True
                                    break

                if global_quit or skip_grasp or label_skip or scene_skip:
                    continue

                grasp_success = contact_lost_step == -1

                if SAVE_TACTILE and not skip_saving_this_grasp:
                    lift_out_path = os.path.join(tactile_out_dir, f"lift_{grasp_idx:05d}.npz")
                    np.savez_compressed(
                        lift_out_path,
                        left_cam=np.array(lift_left_imgs, dtype=np.uint8),
                        right_cam=np.array(lift_right_imgs, dtype=np.uint8),
                        grasp_success=grasp_success,
                        contact_lost_step=contact_lost_step,
                        grasp_label=grasp_label,
                        pose_world=world_se3.to_mat(),
                        joints=joint_tgt,
                    )
                    status = "SUCCESS" if grasp_success else f"FAILED (contact lost @ step {contact_lost_step})"
                    print(f"    -> {status} | saved to {tactile_out_dir}/")

        if viewer is not None:
            viewer.close()

    if VISUALIZE:
        cv2.destroyAllWindows()

    print("\n[Info] All scenes processed.")

if __name__ == "__main__":
    main()