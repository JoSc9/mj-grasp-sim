import time
import numpy as np
import mujoco
import mujoco.viewer


from mgs.util.geo.transforms import SE3Pose
from mgs.gripper.panda_gelsight_mini import GripperPandaGelSightMini 
from mgs.gripper.selector import get_gripper
from omegaconf import OmegaConf


def run_test():
    cfg = OmegaConf.create({"name": "PandaGripperGelsightMini"})
    gripper = get_gripper(cfg)

    # grab the xml snippet and all required stl/obj assets
    try:
        xml_snippet, assets = gripper.to_xml()
    except Exception as e:
        print(f"Failed to grab XML or assets. Did you fix the STL path? Error: {e}")
        return

    # wrap it in a basic mujoco scene
    # turned off gravity so the gripper doesn't just drop to the floor
    scene_xml = f"""
    <mujoco>
        <option gravity="0 0 0" />
        
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 .2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
        </asset>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <geom name="floor" type="plane" size="1 1 0.05" rgba="0.8 0.9 0.8 1" material="matplane"/>
        </worldbody>
        
        {xml_snippet}
    </mujoco>
    """

    # compile the model
    try:
        model = mujoco.MjModel.from_xml_string(scene_xml, assets)
        data = mujoco.MjData(model)
        print("Model compiled successfully!")
    except Exception as e:
        print(f"MuJoCo compilation crashed. Check your XML/STL geometry. Details: {e}")
        return

    print("Launching viewer... (Close the window to stop)")
    
    # start the interactive passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        is_open = True
        last_toggle = time.time()
        
        # set initial control signals (open)
        data.ctrl[0] = 0.04
        data.ctrl[1] = 0.0
        
        while viewer.is_running():
            now = time.time()
            
            # toggle open/close every 2 seconds to check kinematics
            if now - last_toggle > 2.0:
                if is_open:
                    print("Closing fingers...")
                    data.ctrl[0] = 0.0
                    data.ctrl[1] = -0.04
                    is_open = False
                else:
                    print("Opening fingers...")
                    data.ctrl[0] = 0.04
                    data.ctrl[1] = 0.0
                    is_open = True
                
                last_toggle = now

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    run_test()