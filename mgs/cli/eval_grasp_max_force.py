import math
import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.selector import get_gripper
from mgs.obj.selector import get_object
from mgs.sampler.antipodal import AntipodalGraspGenerator
from mgs.sampler.contact import ContactBasedDiff
from mgs.sampler.kin.allegro import AllegroKinematicsModel
from mgs.sampler.kin.dexee import DexeeKinematicsModel
from mgs.sampler.kin.shadow import ShadowKinematicsModel
from mgs.util.const import ASSET_PATH
from mgs.util.file import generate_unique_hash  # uses secrets.token_hex under the hood
from mgs.util.geo.transforms import SE3Pose


def _count_grasps_in_dir(dirpath: str) -> int:
    """Robustly sum #poses across all .npz files in dir."""
    total = 0
    for fn in os.listdir(dirpath):
        if not fn.endswith(".npz"):
            continue
        fpath = os.path.join(dirpath, fn)
        try:
            with np.load(fpath) as z:
                total += int(len(z["poses"]))
        except Exception:
            # ignore any unreadable/partial files (shouldn't happen with atomic rename)
            continue
    return total


def _atomic_save_npz(final_path: str, **arrays):
    """
    Atomic write: save to a temp file in the same directory, then os.replace.
    This avoids readers ever seeing a partially-written file.
    """
    d = os.path.dirname(final_path)
    tmp_name = f".tmp-{os.getpid()}-{generate_unique_hash(8)}.npz"
    tmp_path = os.path.join(d, tmp_name)
    # ensure same filesystem/dir for atomic replace
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, final_path)  # atomic on POSIX when same filesystem


def _fmt_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    seconds = int(round(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h > 99:
        return ">99h"
    return f"{h:02d}:{m:02d}:{s:02d}"


@hydra.main(
    version_base="1.3.2", config_path="config", config_name="gen_gripper_object_grasps"
)
def main(cfg: DictConfig):
    # --- select object ---
    object_id_file = os.path.join(ASSET_PATH, "mj-objects", "fast_eta_objects.txt")
    with open(object_id_file, "r") as file:
        all_object_ids = file.read().splitlines()
        
        
    #object_ids = all_object_ids[int(cfg.object_id)]

    for object_id in all_object_ids:
        # --- components ---
        obj = get_object(object_id)
        assert cfg.gripper.grasp_sampler in [
            "Antipodal",
            "ContactGradient",
        ], "Unsupported sampler"
        sampler = (
            AntipodalGraspGenerator(obj)
            if cfg.gripper.grasp_sampler == "Antipodal"
            else ContactBasedDiff(obj)
        )
        gripper = get_gripper(cfg.gripper)

        print(
            f"Generating grasp candidates using gripper: {cfg.gripper.name}"
            f"\nfor object {object_id} using {cfg.gripper.grasp_sampler} sampler"
        )

        # --- output dir ---
        output_root = os.getenv("MGS_OUTPUT_DIR") or os.path.join(os.getcwd(), "outputs")
        friction_str = "friction_panda"
        output_dir = os.path.join(output_root, "debug_force_eval", cfg.gripper.name, friction_str, object_id)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Setup] Debug mode active. All data will be saved to: {output_dir}")

        # how many are already saved (from all processes)?
        num_total_grasps = _count_grasps_in_dir(output_dir)
        target = int(cfg.target_grasps)
        grasps_to_generate = target - num_total_grasps
        if grasps_to_generate <= 0:
            print("Nothing to do — target already met.")
            continue

        print(f"Grasps to generate (target minus existing): {grasps_to_generate}")

        env = GravitylessObjectGrasping(gripper, obj, render=False)

        # rolling buffers
        buf_poses, buf_joints, buf_forces = [], [],[] 

        # per-process stats
        total_attempts = 0
        total_stable = 0
        attempts_since_last_save = 0
        stable_since_last_save = 0

        # --------- RUNNING STATS & ETA (skipping warmup) ---------
        # configurable, but safe defaults if not present in cfg
        warmup_rounds = int(getattr(cfg, "warmup_rounds", 1))  # skip first N rounds
        ema_alpha = float(getattr(cfg, "eta_ema_alpha", 0.25))  # smoothing for rates/times
        start_time = time.perf_counter()

        # Totals AFTER warmup (used for percentage estimates)
        post_sampled = 0  # number of sampled poses seen
        post_cf = 0  # number of collision-free poses
        post_stable = 0  # number of stable poses

        # Time accumulators AFTER warmup
        post_t_sampling = 0.0  # time spent in sampler.generate_grasps
        post_t_cf = 0.0  # time spent in env.grasp_collision_mask
        post_t_eval = 0.0  # time spent in env.grasp_stability_evaluation_from_joints
        post_t_e2e = 0.0  # end-to-end time measured per "evaluate" cycle

        # EMAs for per-item costs (to be robust to variability)
        ema_t_per_sample = None  # seconds per sampled pose
        ema_t_per_cfcheck = None  # seconds per sampled pose for collision check
        ema_t_per_eval = None  # seconds per CF pose for stability eval
        ema_stable_per_sec = None  # stable grasps per second (local, end-to-end)
        ema_global_grasps_per_sec = None  # global dir growth rate

        # For global rate computation at save times
        last_global_scan_time = time.perf_counter()
        last_global_scan_total = num_total_grasps

        def _ema_update(old, new):
            if old is None:
                return new
            return ema_alpha * new + (1.0 - ema_alpha) * old

        def _print_running_stats(round_index: int):
            # percentages after warmup
            cf_rate = (post_cf / post_sampled) if post_sampled > 0 else float("nan")
            stab_rate = (post_stable / post_cf) if post_cf > 0 else float("nan")

            tps = (
                ema_t_per_sample
                if ema_t_per_sample is not None
                else (post_t_sampling / post_sampled if post_sampled else float("nan"))
            )
            tpc = (
                ema_t_per_cfcheck
                if ema_t_per_cfcheck is not None
                else (post_t_cf / post_sampled if post_sampled else float("nan"))
            )
            tpe = (
                ema_t_per_eval
                if ema_t_per_eval is not None
                else (post_t_eval / post_cf if post_cf else float("nan"))
            )

            local_sps = (
                ema_stable_per_sec
                if ema_stable_per_sec is not None
                else (post_stable / post_t_e2e if post_t_e2e > 0 else float("nan"))
            )

            # ETAs
            now_total = _count_grasps_in_dir(output_dir)
            remaining = max(0, target - now_total)

            local_eta = (
                _fmt_eta(remaining / local_sps)
                if (local_sps and math.isfinite(local_sps) and local_sps > 1e-9)
                else "n/a"
            )
            global_eta = "n/a"
            if ema_global_grasps_per_sec and ema_global_grasps_per_sec > 1e-9:
                global_eta = _fmt_eta(remaining / ema_global_grasps_per_sec)

            print(
                "[STATS] "
                f"round={round_index} | "
                f"CF={cf_rate*100:.1f}% | Stable={stab_rate*100:.1f}% | "
                f"t/sample={tps:.6f}s | t/cf={tpc:.6f}s | t/stab={tpe:.6f}s | "
                f"local stable/s={local_sps:.3f} | "
                f"ETA local={local_eta} | ETA global={global_eta}"
        )

        def maybe_flush(final_flush: bool = False):
            """
            Save a chunk with a random-hash filename. After saving, re-scan the dir to
            see if the global target is already met by any process. Also updates global ETA.
            """
            nonlocal buf_poses, buf_joints, buf_forces
            nonlocal attempts_since_last_save, stable_since_last_save
            nonlocal num_total_grasps, total_stable
            nonlocal ema_global_grasps_per_sec, last_global_scan_time, last_global_scan_total

            if not buf_poses:
                return False

            poses_all = np.concatenate(buf_poses, axis=0)
            joints_all = np.concatenate(buf_joints, axis=0)
            forces_all = np.concatenate(buf_forces, axis=0)

            # decide how many to save now
            if final_flush:
                want = len(poses_all)  # don't waste any grasps on exit
            else:
                want = min(
                    int(cfg.collect_grasps_till_save),
                    len(poses_all),
                )

            if want <= 0:
                return False

            save_poses = poses_all[:want]
            save_joints = joints_all[:want]
            save_forces = forces_all[:want]

            leftovers_poses = poses_all[want:]
            leftovers_joints = joints_all[want:]
            leftovers_forces = forces_all[want:]

            buf_poses = [leftovers_poses] if len(leftovers_poses) else []
            buf_joints = [leftovers_joints] if len(leftovers_joints) else []
            buf_forces = [leftovers_forces] if len(leftovers_forces) else []

            # report success rates (per-process)
            file_sr = (
                float(stable_since_last_save) / float(attempts_since_last_save)
                if attempts_since_last_save > 0
                else float("nan")
            )
            cum_sr = (
                (float(total_stable) / float(total_attempts))
                if total_attempts > 0
                else float("nan")
            )

            fname = f"force_eval_{int(time.time())}-{os.getpid()}-{generate_unique_hash(4)}.npz"
            fpath = os.path.join(output_dir, fname)

            _atomic_save_npz(
                fpath,
                poses=save_poses,
                joints=save_joints,
                max_forces=save_forces, # NEU: Übergabe an die Speicherfunktion
                success_rate_during_gen=np.asarray(file_sr),
            )
            print(
                f"[SAVE DEBUG] {fname} → {len(save_poses)} bewertete Griffe | "
                f"seit letztem Save SR={file_sr*100:.1f}%"
            )

            attempts_since_last_save = 0
            stable_since_last_save = 0

            # re-scan the dir (all processes) to decide whether to continue
            num_total_grasps = _count_grasps_in_dir(output_dir)

            # update global EMA rate based on observed directory growth
            now = time.perf_counter()
            dt = now - last_global_scan_time
            delta = num_total_grasps - last_global_scan_total
            if dt > 0 and delta > 0:
                observed = delta / dt  # grasps per second, global
                ema_global_grasps_per_sec = _ema_update(ema_global_grasps_per_sec, observed)
                last_global_scan_time = now
                last_global_scan_total = num_total_grasps

            if num_total_grasps >= target:
                # we still keep any local leftovers by doing a final flush on exit
                return True  # signal: global target reached
            return False

        round_idx = 0   
        while round_idx < int(cfg.max_rounds):
            # check global progress before starting a new round
            num_total_grasps = _count_grasps_in_dir(output_dir)
            if num_total_grasps >= target:
                break

            round_idx += 1

            # ---- collect collision-free grasps until eval threshold ----
            collected_poses, collected_joints = [], []

            # per-round instrumentation
            r_sampled = 0
            r_cf = 0
            r_stable = 0
            r_t_sampling = 0.0
            r_t_cf = 0.0
            r_t_eval = 0.0

            round_start = time.perf_counter()

            while sum(len(p) for p in collected_poses) < int(cfg.collect_grasps_till_eval):

                # sampling
                t0 = time.perf_counter()
                if cfg.gripper.grasp_sampler == "Antipodal":
                    poses_mat, aux_info = sampler.generate_grasps(
                        num=int(cfg.sample_grasps)  # type: ignore
                    )
                    padding = 0.01
                    joints = gripper.width_to_joints(aux_info["width"] + padding)
                elif cfg.gripper.grasp_sampler == "ContactGradient":
                    all_kins = {
                        "ShadowHand": ShadowKinematicsModel(),
                        "AllegroGripper": AllegroKinematicsModel(),
                        "DexeeGripper": DexeeKinematicsModel(),
                    }
                    kin_model = all_kins[cfg.gripper.name]
                    poses_mat, aux_info = sampler.generate_grasps(  # type: ignore
                        num=int(cfg.sample_grasps), gripper=kin_model  # type: ignore
                    )
                    joints = aux_info["joints"]
                else:
                    raise ValueError("Not known grasp sampler")
                r_t_sampling += time.perf_counter() - t0

                if len(poses_mat) == 0:
                    continue

                r_sampled += int(len(poses_mat))

                # collision check (mask=True means collision-free in this code path)
                poses_se3 = SE3Pose.from_mat(poses_mat)
                t1 = time.perf_counter()
                collision_mask = env.grasp_collision_mask(
                    poses_se3, joints, with_padding=0.002
                )
                r_t_cf += time.perf_counter() - t1

                n_cf_inc = int(np.count_nonzero(collision_mask))
                r_cf += n_cf_inc

                # Debug
                print(f"[Sampling] {r_sampled} grasps generated. {r_cf} collision free. Target: {cfg.collect_grasps_till_eval}", end="\r")

                if n_cf_inc:
                    collected_poses.append(poses_se3.to_mat()[collision_mask])
                    collected_joints.append(joints[collision_mask])

            if not collected_poses:
                continue

            cf_poses_mat = np.concatenate(collected_poses, axis=0)
            cf_joints = np.concatenate(collected_joints, axis=0)
            cf_poses = SE3Pose.from_mat(cf_poses_mat)

            # ---- Max impulse force evaluation ----
            t2 = time.perf_counter()
            max_forces = env.grasp_max_impulse_force_evaluation(
                cf_poses, cf_joints, start_force=10.0, max_force=300.0, force_step=10.0
            )
            r_t_eval += time.perf_counter() - t2

            attempts = len(cf_poses)

            # Filter grasps that failed immediately during closing process 
            survived_mask = np.sum(max_forces, axis=1) > 0
            stables = int(np.count_nonzero(survived_mask))

            total_attempts += attempts
            attempts_since_last_save += attempts
            total_stable += stables
            stable_since_last_save += stables
            r_stable += stables

            # update post-warmup running stats
            if round_idx > warmup_rounds:
                post_sampled += r_sampled
                post_cf += r_cf
                post_stable += r_stable
                post_t_sampling += r_t_sampling
                post_t_cf += r_t_cf
                post_t_eval += r_t_eval

                # end-to-end time for this evaluation cycle
                r_t_e2e = time.perf_counter() - round_start
                post_t_e2e += r_t_e2e

                # per-item costs and throughput (EMA)
                if r_sampled > 0:
                    ema_t_per_sample = _ema_update(
                        ema_t_per_sample, r_t_sampling / float(r_sampled)
                    )
                    ema_t_per_cfcheck = _ema_update(
                        ema_t_per_cfcheck, r_t_cf / float(r_sampled)
                    )
                if attempts > 0:
                    ema_t_per_eval = _ema_update(ema_t_per_eval, r_t_eval / float(attempts))
                if r_t_e2e > 0:
                    ema_stable_per_sec = _ema_update(ema_stable_per_sec, r_stable / r_t_e2e)

                _print_running_stats(round_idx)

            if stables:
                buf_poses.append(cf_poses.to_mat()[survived_mask])
                buf_joints.append(cf_joints[survived_mask])
                buf_forces.append(max_forces[survived_mask])

            # save if chunk ready
            if sum(len(p) for p in buf_poses) >= int(cfg.collect_grasps_till_save):
                stop = maybe_flush(final_flush=False)
                if stop:
                    break

        # final flush of leftovers so no grasps are wasted
        if sum(len(p) for p in buf_poses):
            _ = maybe_flush(final_flush=True)

        print("Done!")


if __name__ == "__main__":
    main()
