import mujoco
import numpy as np
import time
from mgs.gripper.selector import get_gripper
from omegaconf import OmegaConf

"""
Script to determine the base to contact transform:
    - Loads gripper and yellow box between fingers
    - Close the gripper
    - Stop closing process when contact is detected
"""

# Config setup
GRIPPER_NAME = "PandaGripperGelsightMini" 
HAND_BODY_NAME = "hand"                    


def main():
    print(f"[Info] Loading gripper '{GRIPPER_NAME}' for calibration...")

    # Load gripper
    try:
        cfg = OmegaConf.create({"name": GRIPPER_NAME})
        gripper = get_gripper(cfg)
        xml, assets = gripper.to_xml()
    except Exception as e:
        print(f"[Error] Could not load gripper. Check the name '{GRIPPER_NAME}'.")
        print(f"Details: {e}")
        return

    # Create a minimal gravity-free world
    full_xml = f"""
    <mujoco>
        <option gravity="0 0 0" />
        <worldbody>
            <body name="target_box" pos="0 0 0.1">
                <geom name="box" type="box" size="0.01 0.01 0.01" rgba="1 1 0 1" contype="1" conaffinity="1" mass="0.01"/>
            </body>
        </worldbody>
        {xml}
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(full_xml, assets)
    data = mujoco.MjData(model)


    # Start viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        mujoco.mj_resetData(model, data)

        finger1_id = model.joint("finger_joint1").qposadr
        finger2_id = model.joint("finger_joint2").qposadr

        # Open gripper
        data.qpos[finger1_id] = 0.04
        data.qpos[finger2_id] = 0.0

        mujoco.mj_forward(model, data)
        viewer.sync()
        # Wait to open mujoco viewer
        time.sleep(1.0)

        # Run simulation: closing the gripper
        print("[Info] Simulating gripper closing to detect self-collision...")

        start_width = 0.0
        target_width = 0.04

        steps = 2000

        contact_found = False
        avg_contact_point = None

        box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
        
        for i in range(steps):
            # Adapt crtl value slowly
            alpha = i / steps
            current_target = (1 - alpha) * start_width + alpha * target_width
            
            if model.nu > 0:
                # Actuator 0 (range 0-0.04)
                data.ctrl[0] = current_target

                # Actuator 1 (range -0.04-0)
                # Equal to 0.04 translation of first actuator
                data.ctrl[1] = current_target - 0.04

            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Check if any collision occurred
            if data.ncon > 0:
                # Filter contacts to find actual touches (negative distance = penetration)
                # We look for contacts with small distance or penetration
                valid_points = []
                is_box_contact = False
                for c_i in range(data.ncon):
                    contact = data.contact[c_i]
                    if contact.geom1 == box_id or contact.geom2 == box_id:
                        valid_points.append(contact.pos)
                        is_box_contact = True
                
                if is_box_contact:
                    # Calculate the center of all contact points
                    avg_contact_point = np.mean(valid_points, axis=0)
                    contact_found = True
                    print(f"[Success] Collision detected at step {i}!")
                    break


        if not contact_found:
            print("\n[Error] Could not determine contact point.")
            return

        # Calculate Transform (World -> Local Hand Frame)
        try:
            id_hand = model.body(HAND_BODY_NAME).id
        except KeyError:
            print(f"\n[Error] Body '{HAND_BODY_NAME}' not found in the model.")
            print("Available bodies:")
            for i in range(model.nbody):
                print(f" - {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")
            return

        pos_hand = data.xpos[id_hand]               # World position of hand base
        mat_hand = data.xmat[id_hand].reshape(3, 3) # World orientation of hand base

        # Vector from Hand to Contact (in World Frame)
        vec_world = avg_contact_point - pos_hand

        # Transform to Local Hand Frame: v_local = R_hand^T * v_world
        vec_local = mat_hand.T @ vec_world

        # Output Results
        print("\n" + "="*60)
        print(" CALIBRATION RESULTS ")
        print("="*60)
        print(f"Hand Base Position (World): {pos_hand}")
        print(f"Sensor Contact Point (World): {avg_contact_point}")
        print("-" * 60)
        print("base_to_contact_transform:")
        print("")
        print(f"np.array([{vec_local[0]:.5f}, {vec_local[1]:.5f}, {vec_local[2]:.5f}])")
        print("")
        print("-" * 60)
        print("Interpretation:")
        print(f"X (Height): {vec_local[0]:.5f}")
        print(f"Y (Side):   {vec_local[1]:.5f}")
        print(f"Z (Depth):  {vec_local[2]:.5f}")
        print("="*60)

        input("\n[Info] Press ENTER to close the viewer...")

        
if __name__ == "__main__":
    main()