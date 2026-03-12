import time
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf
from mgs.gripper.selector import get_gripper

GRIPPER_NAME = "PandaGripperGelsightMini"

# Closed joint position panda gripper 
CLOSED_QPOS_1 = 0.0
CLOSED_QPOS_2 = -0.04

# Name of the collision geoms to detect 
GEOM_NAME_LEFT = "col_gelsight_left"
GEOM_NAME_RIGHT = "col_gelsight_right"

def measure_thickness():
    """
    - Load gripper
    - Closes gripper until the sensors collide
    - Calculate the sensor thickness based on the joint positions at collision
    """

    print(f"[Info] Loading gripper '{GRIPPER_NAME}'.")

    # Load the Gripper Model
    try:
        cfg = OmegaConf.create({"name": GRIPPER_NAME})
        gripper = get_gripper(cfg)
        xml_gripper, assets = gripper.to_xml()
    except Exception as e:
        print(f"[Error] Failed to load gripper: {e}")
        return

    # Create a minimal simulation environment (No gravity, no floor)
    full_xml = f"""
    <mujoco>
        <option gravity="0 0 0" />
        <worldbody>
        </worldbody>
        {xml_gripper}
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(full_xml, assets)
    data = mujoco.MjData(model)

    # Get IDs for Joints and Geoms
    try:
        # Joints
        id_joint1 = model.joint("finger_joint1").qposadr
        id_joint2 = model.joint("finger_joint2").qposadr
        
        # Geoms (The sensor pads)
        id_geom_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME_LEFT)
        id_geom_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME_RIGHT)
        
        if id_geom_left == -1 or id_geom_right == -1:
            raise ValueError(f"Could not find geoms '{GEOM_NAME_LEFT}' or '{GEOM_NAME_RIGHT}'")
            
    except Exception as e:
        print(f"[Error] Model structure mismatch: {e}")
        return

    # Initialize Simulation
    mujoco.mj_resetData(model, data)
    
    # Open gripper
    data.qpos[id_joint1] = 0.04
    data.qpos[id_joint2] = 0.0   
    
    mujoco.mj_forward(model, data)

    print("[Info] Closing gripper...")

    # Simulation Loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable contact visualization
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        
        # Simulation parameters
        steps = 3000
        detected = False
        
        
        for i in range(steps):
            
            # Linear interpolation between open and closed
            alpha = i / steps
            
            # Target for Finger 1: 0.04 -> 0.0
            target_1 = (1 - alpha) * 0.04 + alpha * CLOSED_QPOS_1
            
            # Target for Finger 2: 0.0 -> -0.04
            target_2 = (1 - alpha) * 0.0 + alpha * CLOSED_QPOS_2
            
            # Apply Control
            if model.nu >= 2:
                data.ctrl[0] = target_1
                data.ctrl[1] = target_2
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Check for specific collision
            if data.ncon > 0:
                for c_i in range(data.ncon):
                    contact = data.contact[c_i]
                    
                    # Check if the collision is between our two sensor pads
                    g1 = contact.geom1
                    g2 = contact.geom2
                    
                    is_sensor_contact = (
                        (g1 == id_geom_left and g2 == id_geom_right) or
                        (g1 == id_geom_right and g2 == id_geom_left)
                    )
                    
                    if is_sensor_contact:
                        detected = True
                        
                        current_q1 = float(data.qpos[id_joint1])
                        current_q2 = float(data.qpos[id_joint2])
                        
                        # For Finger 1: Distance = Current - 0.0
                        offset_1 = abs(current_q1 - CLOSED_QPOS_1)
                        
                        # For Finger 2: Distance = Current - (-0.04)
                        offset_2 = abs(current_q2 - CLOSED_QPOS_2)
                        
                        total_width_at_collision = offset_1 + offset_2
                        thickness_per_pad = total_width_at_collision / 2.0
                        
                        print("\n" + "="*50)
                        print("  MEASUREMENT SUCCESSFUL")
                        print("="*50)
                        print(f"Collision detected at step {i}")
                        print(f"Joint 1 Position: {current_q1:.5f} (Dist from closed: {offset_1:.5f})")
                        print(f"Joint 2 Position: {current_q2:.5f} (Dist from closed: {offset_2:.5f})")
                        print("-" * 50)
                        print(f"Total width occupied by sensors:  {total_width_at_collision:.5f} m")
                        print(f"Estimated THICKNESS per sensor:   {thickness_per_pad:.5f} m")
                        print("="*50)
                       
                        # Exit function on success
                        return 
                    
        if not detected:
            print("[Error] Fingers closed fully but no collision detected between sensors.")
         
if __name__ == "__main__":
    measure_thickness()