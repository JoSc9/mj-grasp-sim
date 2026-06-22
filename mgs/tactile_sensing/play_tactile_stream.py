import os
import glob
import numpy as np
import cv2

def play_tactile_stream(npz_path, playback_speed_fps=30):
    """Loads a tactile .npz file and plays it back as a video with metadata."""
    if not os.path.exists(npz_path):
        print(f"[Error] File not found: {npz_path}")
        return False

    print(f"\n--- Loading: {os.path.basename(npz_path)} ---")
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract image arrays
    left_frames = data['left_cam']
    right_frames = data['right_cam']
    num_frames = len(left_frames)
    
    # Extract metadata safely (using .get() since grasp and lift files have different keys)
    contact_frame_idx = data.get('contact_frame_idx', -1)
    grasp_label = data.get('grasp_label', 'unknown')
    grasp_success = data.get('grasp_success', None)
    
    print(f"Frames: {num_frames} | Label: {grasp_label}")
    
    delay = int(1000 / playback_speed_fps)

    print("Controls:")
    print(" [ space ] - Pause / Play")
    print(" [ n ]     - Skip to next file")
    print(" [ q ]     - Quit entirely\n")

    for i in range(num_frames):
        # Convert RGB to BGR for OpenCV display
        img_l = cv2.cvtColor(left_frames[i], cv2.COLOR_RGB2BGR)
        img_r = cv2.cvtColor(right_frames[i], cv2.COLOR_RGB2BGR)
        
        # --- Draw Overlays ---
        # 1. Base Info
        cv2.putText(img_l, f"Frame: {i}/{num_frames - 1}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_l, f"Label: {grasp_label}", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # 2. Grasp Phase Specifics (Contact)
        if "grasp" in os.path.basename(npz_path):
            if contact_frame_idx != -1 and i >= contact_frame_idx:
                cv2.putText(img_r, "CONTACT", (10, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. Lift Phase Specifics (Success/Fail)
        if "lift" in os.path.basename(npz_path):
            if grasp_success is not None:
                status_text = "SUCCESS" if grasp_success else "FAILED (Dropped)"
                status_color = (0, 255, 0) if grasp_success else (0, 0, 255)
                cv2.putText(img_r, status_text, (10, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Stitch horizontally
        combined_view = np.hstack((img_l, img_r))
        
        cv2.imshow("Tactile Playback (Left | Right)", combined_view)
        
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            return "quit"
        elif key == ord('n'):
            return "next"
        elif key == ord(' '):
            # Pause execution until space is pressed again
            while True:
                pause_key = cv2.waitKey(0) & 0xFF
                if pause_key == ord(' '):
                    break
                elif pause_key == ord('q'):
                    return "quit"
                elif pause_key == ord('n'):
                    return "next"
                    
    # Automatically pause at the end of the clip so it doesn't just vanish
    while True:
        end_key = cv2.waitKey(0) & 0xFF
        if end_key == ord(' '):
            return "next" # Treat space at the end of a video as "next"
        elif end_key == ord('n'):
            return "next"
        elif end_key == ord('q'):
            return "quit"

if __name__ == "__main__":  
    # 1. Broad search pattern to find all tactile feedback folders
    # scenes / <gripper> / <scene_hash> / tactile_feedback / <object_stem>
    base_out_dir = os.getenv("MGS_OUTPUT_DIR")
    feedback_dir_pattern = os.path.join(base_out_dir, "02_scene", "*", "*", "tactile_feedback", "*")
    feedback_dirs = glob.glob(feedback_dir_pattern)
    feedback_dirs.sort()  # This ensures we process scene-by-scene cleanly

    if not feedback_dirs:
        print(f"[Error] No tactile feedback directories found.")
    else:
        print(f"Found {len(feedback_dirs)} unique object/scene directories.")
        print("\nWindow Controls:\n [ space ] - Advance from Grasp to Lift / Pause\n [ n ]     - Skip entire Grasp-Lift pair\n [ q ]     - Quit entirely")

        global_quit = False

        # Loop through each scene/object directory one by one
        for target_dir in feedback_dirs:
            if global_quit:
                break

            # Find all .npz files inside this SPECIFIC scene directory
            npz_files = glob.glob(os.path.join(target_dir, "*.npz"))
            if not npz_files:
                continue

            # Separate into grasp/lift pairs for this scene
            grasps = {}
            lifts = {}
            for fpath in npz_files:
                fname = os.path.basename(fpath)
                prefix, idx_ext = fname.split('_')
                idx = idx_ext.split('.')[0]
                
                if prefix == "grasp":
                    grasps[idx] = fpath
                elif prefix == "lift":
                    lifts[idx] = fpath

            # Sort the indices (000, 001, 002...) to guarantee grasp-lift pairing order
            sorted_indices = sorted(grasps.keys())
            
            print("\n" + "="*70)
            print(f"PROCESSING SCENE DIRECTORY: ...{target_dir[-50:]}")
            print(f"Found {len(sorted_indices)} paired trials in this scene.")
            print("="*70)

            for idx in sorted_indices:
                grasp_file = grasps[idx]
                lift_file = lifts.get(idx, None)
                
                print(f"\n--- Sequence Pair [{idx}] ---")
                
                # 1. Play Grasp Phase (Forces pause at end of grasp)
                action = play_tactile_stream(grasp_file, playback_speed_fps=30)
                
                if action == "quit":
                    global_quit = True
                    break
                elif action == "next_pair":
                    continue  # Skips directly to the next index in this scene
                    
                # 2. Play corresponding Lift Phase if it exists
                if lift_file:
                    action = play_tactile_stream(lift_file, playback_speed_fps=30)
                    if action == "quit":
                        global_quit = True
                        break
                    # If 'next_pair', it naturally falls through to the next loop iteration
                else:
                    print(f"  [Info] No matching lift file found for grasp index {idx}.")

        cv2.destroyAllWindows()
        print("\n[Info] Playback finished.")