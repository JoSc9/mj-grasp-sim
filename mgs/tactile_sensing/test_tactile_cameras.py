import time
import cv2
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf

from mgs.gripper.selector import get_gripper
from mgs.sensors.gelsight_mini.gelsight_mini import GelSightMini

GRIPPER_NAME = "PandaGripperGelsightMini"
USE_MUJOCO_VIEWER = True

class DummyArgs:
    save_dir = "./"
    cam_width = 640       
    cam_height = 480      

dummy_args = DummyArgs()

def main():
    print("[Info] Setting up the simulation environment...")

    # Load gripper
    try:
        cfg = OmegaConf.create({"name": GRIPPER_NAME})
        gripper = get_gripper(cfg)
        xml_gripper, assets = gripper.to_xml()
    except Exception as e:
        print(f"[Error] Failed to load gripper: {e}")
        return

    # Create simple world with no gravity
    full_xml = f"""
    <mujoco>
        <option gravity="0 0 0" />
        <worldbody>
            <light pos="0 0 1"/>
            <body name="test_object" pos="0 0 0.0984">
                <geom name="box_geom" type="sphere" size="0.007" rgba="1 0 0 1" mass="0.05" condim="6"/>
            </body>
        </worldbody>
        {xml_gripper}
    </mujoco>
    """

    # Compile model
    model = mujoco.MjModel.from_xml_string(full_xml, assets)
    data = mujoco.MjData(model)

    # Initialize sensors
    print("[Info] Initializing gelsight mini camera sensor...")
    try:
        left_sensor = GelSightMini(
            args=dummy_args, 
            model=model, 
            data=data, 
            cam_name="tactile_cam_left"
        )
    except Exception as e:
        print(f"[Error] GelSight init failed. Error: {e}")
        return

    # Reset data
    mujoco.mj_resetData(model, data)

    id_joint1 = model.joint("finger_joint1").qposadr
    id_joint2 = model.joint("finger_joint2").qposadr

    # Open gripper
    data.qpos[id_joint1] = 0.04
    data.qpos[id_joint2] = 0.0
    mujoco.mj_forward(model, data)

    print("[Info] Closing gripper...")

    # Close gripper
    target_joints = gripper.width_to_joints(0.0)
    
    if model.nu >= 2:
        data.ctrl[0] = target_joints[0]
        data.ctrl[1] = target_joints[1]

    # Config simulation
    steps = 1000  
    # Render every 15th step
    render_interval = 2 

    # Initialize viewer
    viewer = None
    if USE_MUJOCO_VIEWER:
        viewer = mujoco.viewer.launch_passive(model, data)

    try:
        for i in range(steps):
            mujoco.mj_step(model, data)
            if viewer is not None and viewer.is_running():
                viewer.sync()
            
            current_q1 = data.qpos[id_joint1]
            print(f"Step {i}: Joint position finger 1 = {current_q1.item():.6f}")

            force_left = data.actuator_force[0]
            force_right = data.actuator_force[1]
            print(f"Step{i}: Motor Force Left = {force_left:.4f} N | Right = {force_right:.4f} N")

            total_force = 0
            for n in range(data.ncon):
                contact = data.contact[n]
                # Check if collision with hard_stop_left or hard_stop_right occurs
                geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                if "hard_stop" in str(geom1_name) or "hard_stop" in str(geom2_name):
                    # Calculate penetration depth of hard_stop box
                    print(f"Hard stop reached. Penetration depth: {contact.dist:.6f}")


            # Render camera image
            if i % render_interval == 0:
                tactile_img_rgb = left_sensor.tactile_image
                if tactile_img_rgb is not None:
                    # Convert for OpenCV
                    tactile_img_bgr = cv2.cvtColor(tactile_img_rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
                    tactile_img_bgr = cv2.normalize(tactile_img_bgr, None, 0, 255, cv2.NORM_MINMAX)
                    tactile_img_bgr = np.uint8(tactile_img_bgr)

                    # Update tactile image
                    cv2.imshow("Live Tactile Image - Left Finger", tactile_img_bgr)

                    # Stop stream if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[Info] Stream stopped manually.")
                        break

        print("[Info] Simulation finished. Press Enter in terminal to close all windows...")
        input()
    finally:
        # Close all windwos
        cv2.destroyAllWindows()
        if viewer is not None:
            viewer.close()

if __name__ == "__main__":
    main()