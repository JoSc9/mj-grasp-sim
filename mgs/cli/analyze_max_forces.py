import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def load_debug_data(debug_dir: str):
    """
    Loads all .npz files from the debug directory and concatenates 
    the 'max_forces' matrices into a single numpy array.
    """
    all_forces = []
    
    if not os.path.exists(debug_dir):
        print(f"[Error] Directory not found: {debug_dir}")
        return None

    for fn in os.listdir(debug_dir):
        if fn.endswith(".npz"):
            try:
                with np.load(os.path.join(debug_dir, fn)) as data:
                    all_forces.append(data["max_forces"])
            except Exception as e:
                print(f"[Warning] Error loading {fn}: {e}")

    if not all_forces:
        print("[Error] No valid data found in the specified directory.")
        return None

    # Concatenate all loaded matrices along the first axis (number of grasps)
    forces = np.concatenate(all_forces, axis=0) # Shape: (N_grasps, 6)
    return forces

def plot_grasp_analytics(forces: np.ndarray, plot_title: str, save_path: str = None):
    """
    Generates statistical plots for the evaluated grasp forces,
    including a directional bias bar chart and a worst-case histogram.
    """
    num_grasps = forces.shape[0]
    dir_names = ["+X", "+Y", "+Z", "-X", "-Y", "-Z"]
    
    print(f"[Info] Analyzing {num_grasps} grasps...")

    # Statistical calculations
    mean_forces = np.mean(forces, axis=0)
    std_forces = np.std(forces, axis=0)
    
    # Calculate the "weakest link" for each grasp (the minimum force it could withstand)
    min_force_per_grasp = np.min(forces, axis=1) 
    
    # --- Figure Setup ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(plot_title, fontsize=16)

    # --- Plot 1: Directional Bias (Bar Chart) ---
    ax1 = axes[0]
    bars = ax1.bar(dir_names, mean_forces, yerr=std_forces, capsize=5, color='skyblue', edgecolor='black')
    ax1.set_title('Average Holding Force per Direction')
    ax1.set_ylabel('Maximum Force (Newtons)')
    ax1.set_xlabel('Impulse / Kick Direction')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate values above the bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.1f}N', ha='center', va='bottom', fontsize=10)

    # --- Plot 2: Worst-Case Distribution (Histogram) ---
    ax2 = axes[1]
    ax2.hist(min_force_per_grasp, bins=20, color='coral', edgecolor='black', alpha=0.8)
    ax2.set_title('Distribution of the "Weakest Links"\n(Minimum Holding Force per Grasp)')
    ax2.set_ylabel('Number of Grasps')
    ax2.set_xlabel('Minimum Holding Force (Newtons)')
    
    # Add a vertical line for the mean of the minimum forces
    mean_min_force = np.mean(min_force_per_grasp)
    ax2.axvline(mean_min_force, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_min_force:.1f}N')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Success] Plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    input_root = os.path.expanduser(os.path.join("~/mj_data", "out", "debug_force_eval", "PandaGripperGelsightMini", "friction_gelsight_mini"))
    results_base_dir = os.path.join(os.path.dirname(input_root), "results_eval")
    os.makedirs(results_base_dir, exist_ok=True)

    category_name = os.path.basename(input_root)
    print(results_base_dir)

    if not os.path.exists(input_root):
        print(f"[Error] Path doesn't exist: {input_root}")
    else:
        object_folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
        
        for object_id in sorted(object_folders):
            object_path = os.path.join(input_root, object_id)
            
            # Load data
            forces_data = load_debug_data(object_path)
            
            if forces_data is not None:
                plot_title = f"Grasp Evaluation: {object_id}\nSource: {category_name}"
                file_name = f"{category_name}_{object_id}.png"
                save_path = os.path.join(results_base_dir, file_name)
                
                plot_grasp_analytics(forces_data, plot_title, save_path=save_path)
            else:
                print(f"No data found for object, skip object: {object_id}")



