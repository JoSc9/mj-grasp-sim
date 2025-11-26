import os
from copy import deepcopy

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from mgs.env.selector import get_env_from_dict
from mgs.gripper.base import MjScannableGripper
from mgs.gripper.selector import get_gripper
from mgs.sampler.helper import farthest_point_sampling
from mgs.util.img_proc import detect_outlier, rgbd_to_pcd, voxel_downsample_pcd


def scan(cfg: DictConfig, scene_def):
    gripper = get_gripper(cfg.gripper)
    assert isinstance(gripper, MjScannableGripper)
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)))
    images, extrinsics, image_masks, _ = env.scan(num_images=cfg.num_images)
    intrinsics = env.get_camera_intrinsics()
    return images, extrinsics, intrinsics, image_masks


@hydra.main(config_path="config", config_name="render_scene")
def main(cfg: DictConfig):
    output_dir_all = os.getenv("MGS_OUTPUT_DIR")
    input_dir_all = os.getenv("MGS_INPUT_DIR")

    assert output_dir_all is not None
    assert input_dir_all is not None
    input_dir_all = os.path.join(input_dir_all, cfg.gripper.name)
    scene_dir_list = [
        d for d in os.listdir(input_dir_all) if os.path.isdir(os.path.join(input_dir_all, d))
    ]

    num  = len(scene_dir_list)
    count = 1
    for scene_dir in scene_dir_list:

        input_dir = os.path.join(input_dir_all, scene_dir)
        print("Scene dir: ", input_dir)

        scene_path = os.path.join(input_dir, "scene.npz")
        scene = np.load(scene_path, allow_pickle=True)
        scene_dict = scene["scene_definition"].item()
        images, extrinsics, intrinsics, image_masks = scan(
            deepcopy(cfg), deepcopy(scene_dict)
        )
        pcd, feature = rgbd_to_pcd(images, intrinsics, extrinsics)
        pcd = pcd[image_masks]
        feature = feature[image_masks]

        region_mask = np.all(
            (pcd < np.array([[0.225, 0.225, 1.0]]))
            & (pcd > np.array([[-0.225, -0.225, -0.01]])),
            axis=-1,
        )
        pcd = pcd[region_mask]
        feature = feature[region_mask]

        pcd, feature = voxel_downsample_pcd(pcd, feature, voxel_size=0.002)
        mask = detect_outlier(pcd, radius=0.008, min_neighbors=2)
        pcd, feature = pcd[mask], feature[mask]
        idx = farthest_point_sampling(
            jnp.asarray(pcd, dtype=jnp.float32), num_samples=15000
        )
        pcd = pcd[idx]
        feature = feature[idx]

        output_dir = os.path.join(output_dir_all, cfg.gripper.name, scene_dir)
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, "scene_pcd"),
            **{
                "points": np.asarray(pcd, dtype=np.float32),
                "colors": np.asarray(feature, dtype=np.float32),
            },
        )
        print(f"Finished with scene {count} of {num}!")
        count += 1


if __name__ == "__main__":
    main()
