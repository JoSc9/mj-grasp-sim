import os
import glob
import cv2
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf

from mgs.env.clutter_table import ClutterTableEnv
from mgs.gripper.selector import get_gripper
from mgs.obj.selector import get_object
from mgs.sensors.gelsight_mini.gelsight_mini import GelSightMini
from mgs.util.geo.transforms import SE3Pose

class DummyArgs:
    save_dir = "./"
    cam_width = 640       
    cam_height = 480      

def main():
    # 1. Configuration 
    GRIPPER_NAME = "PandaGripperGelsightMini"
    # Replace with your target object ID
    OBJECT_ID = "011_banana" 
    
    BASE_DIR = os.path.expanduser(f"~/mj_data/out/debug_force_eval/{GRIPPER_NAME}/friction_gelsight_mini/{OBJECT_ID}")
    
    # 2. Find all .npz files in the directory
    if not os.path.exists(BASE_DIR):
        print(f"[Error] Directory not found: {BASE_DIR}")
        return

    # Grab all .npz files
    npz_files = glob.glob(os.path.join(BASE_DIR, "*.npz"))
    
    if not npz_files:
        print(f"[Warning] No .npz files found in {BASE_DIR}")
        return

    print(f"[Info] Found {len(npz_files)} .npz files for object {OBJECT_ID}.")

    # 3. Setup MuJoCo Environment
    print("[Info] Setting up environment...")
    cfg = OmegaConf.create({"name": GRIPPER_NAME})
    gripper = get_gripper(cfg)
    obj = get_object(OBJECT_ID)
    
    # Set up table environment
    env = ClutterTableEnv(gripper,[obj], headless=True, scene_randomization=False)
    model = env.model
    model.vis.map.znear = 0.001
    model.vis.map.zfar = 50.0
    mj_data = env.data

    # Drop object and save resting state
    print("[INFO] Generating clutter and letting object settle")
    # Hide gripper far away before spawning so it doesn't bat the object mid-air
    hide_pose = SE3Pose(np.array([10.0, 10.0, 10.0]), np.array([1.0, 0.0, 0.0, 0.0]), type="wxyz")
    gripper.set_pose(env, hide_pose)
    mujoco.mj_forward(model, mj_data)

    # Drop objects and simulate until it stops moving
    env.gen_clutter()

    # Save physics state to reload it for each grasp
    settled_state = env.get_state()

    # 4. Initialize GelSight Sensors
    dummy_args = DummyArgs()
    print("[Info] Initializing GelSight sensors...")
    left_sensor = GelSightMini(args=dummy_args, model=model, data=mj_data, cam_name="tactile_cam_left")   
    right_sensor = GelSightMini(args=dummy_args, model=model, data=mj_data, cam_name="tactile_cam_right")

    # 5. Launch Viewer
    viewer = mujoco.viewer.launch_passive(model, mj_data)
    viewer.opt.geomgroup[4] = 1  # Show GelSight shell

    print("\n=======================================================")
    print("CONTROLS:")
    print(" [ space ] - Next grasp")
    print(" [ n ]     - Skip to next .npz file")
    print(" [ q ]     - Quit")
    print("=======================================================\n")

    try:
        # Outer Loop: Iterate through files
        for file_idx, npz_path in enumerate(npz_files):
            file_name = os.path.basename(npz_path)
            print(f"\n>>> Loading File {file_idx + 1}/{len(npz_files)}: {file_name}")
            
            try:
                data = np.load(npz_path)
                poses_mat = data["poses"]   
                joints_target = data["joints"] 
            except Exception as e:
                print(f"[Error] Could not read {file_name}: {e}")
                continue
            
            print("[INFO] Filtering collisions for the file ")
            env.set_state(settled_state)
            mujoco.mj_forward(model, mj_data)

            # Get the Object-to-World Transform (o2w) ---
            try:
                jnt_adr_start = model.jnt(f"{obj.name}:joint").qposadr[0].item()
                obj_pos = mj_data.qpos[jnt_adr_start : jnt_adr_start + 3]
                obj_quat = mj_data.qpos[jnt_adr_start + 3 : jnt_adr_start + 7]
                o2w = SE3Pose(np.copy(obj_pos), np.copy(obj_quat), type="wxyz")
            except Exception as e:
                print(f"[Warning] Could not find joint for {obj.name}. Defaulting to origin.")
                o2w = SE3Pose(np.array([0., 0., 0.]), np.array([1., 0., 0., 0.]), type="wxyz")

            # Calculate Final Base Pose 
            raw_se3_pose = SE3Pose.from_mat(poses_mat)
            world_se3_pose = o2w @ raw_se3_pose 

            collision_free_mask = env.grasp_collision_mask(world_se3_pose, joints_target, with_padding=0.002)

            valid_poses_mat = poses_mat[collision_free_mask]
            valid_joints_target = joints_target[collision_free_mask]

            print(f"[INFO] Kept {len(valid_poses_mat)} / {len(poses_mat)} collision-free grasps")  

            if len(valid_poses_mat) == 0:
                print("[INFO] Skipping file (no valid grasps) ")
                continue
            
            skip_file = False

            # Inner Loop: Iterate through grasps in the current file
            for grasp_idx, (poses_mat, joint_tgt) in enumerate(zip(valid_poses_mat, valid_joints_target)):
                print(f"--- Visualizing Grasp {grasp_idx + 1}/{len(valid_poses_mat)} (File: {file_name}) ---")
                
                # Reload saved table state
                env.set_state(settled_state)
                # Update kinematics to read accurate objec position
                mujoco.mj_forward(model, mj_data) 
                
                # Get the Object-to-World Transform (o2w) ---
                try:
                    jnt_adr_start = model.jnt(f"{obj.name}:joint").qposadr[0].item()
                    obj_pos = mj_data.qpos[jnt_adr_start : jnt_adr_start + 3]
                    obj_quat = mj_data.qpos[jnt_adr_start + 3 : jnt_adr_start + 7]
                    o2w = SE3Pose(np.copy(obj_pos), np.copy(obj_quat), type="wxyz")
                except Exception as e:
                    print(f"[Warning] Could not find joint for {obj.name}. Defaulting to origin.")
                    o2w = SE3Pose(np.array([0., 0., 0.]), np.array([1., 0., 0., 0.]), type="wxyz")

                # Calculate Final Base Pose 
                raw_se3_pose = SE3Pose.from_mat(poses_mat)
                b2c = gripper.base_to_contact_transform()
                
                # Convert: Object Frame -> World Frame -> Base Offset
                world_se3_pose = o2w @ raw_se3_pose
                pose_processed = world_se3_pose @ b2c
                
                # Instantly Teleport Gripper
                gripper.open_gripper(env)
                gripper.set_pose(env, pose_processed)
                mujoco.mj_forward(model, mj_data)

                # Close gripper
                close_ctrl = gripper.width_to_joints(-1.0)
                mj_data.ctrl[:] = close_ctrl 
                

                # Step simulation to watch it close
                steps_to_close = 1000
                render_interval = 5
                skip_to_next_grasp = False

                for step in range(steps_to_close):
                    mujoco.mj_step(model, mj_data)
                    
                    if viewer.is_running():
                        viewer.sync()

                    # Render tactile images periodically
                    if step % render_interval == 0:
                        img_l = left_sensor.tactile_image
                        img_r = right_sensor.tactile_image

                        if img_l is not None and img_r is not None:
                            bgr_l = np.uint8(cv2.normalize(cv2.cvtColor(img_l.astype(np.float32), cv2.COLOR_RGB2BGR), None, 0, 255, cv2.NORM_MINMAX))
                            bgr_r = np.uint8(cv2.normalize(cv2.cvtColor(img_r.astype(np.float32), cv2.COLOR_RGB2BGR), None, 0, 255, cv2.NORM_MINMAX))

                            cv2.imshow(f"Left Sensor", bgr_l)
                            cv2.imshow(f"Right Sensor", bgr_r)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("Quitting visualization.")
                            return
                        elif key == ord(' '):
                            print("Skipping to next grasp...")
                            skip_to_next_grasp = True
                            break
                        elif key == ord('n'):
                            print("Skipping to next file...")
                            skip_file = True
                            break

                if skip_file:
                    break # Break out of the inner loop, moving to the next file in the outer loop
                
                if skip_to_next_grasp:
                    continue
                
                # Wait for user at the end of the grasp
                print("Grasp complete. Press [Space] for next grasp, [n] for next file, [q] to quit...")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord(' '):
                        break
                    elif key == ord('n'):
                        skip_file = True
                        break
                    elif key == ord('q'):
                        return

            if skip_file:
                continue

    finally:
        cv2.destroyAllWindows()
        if viewer is not None:
            viewer.close()
        print("\n[Info] Visualization finished.")

if __name__ == "__main__":
    main()