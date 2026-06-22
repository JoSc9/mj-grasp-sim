# Tactile Sensor Integration — GelSight Mini

This document describes the tactile sensor integration added on top of the base
`mj-grasp-sim` pipeline. It covers the gripper configuration, sensor
initialization, output data structure, and how to run the extended pipeline.

---

## Overview

The standard Panda gripper has been extended with two **GelSight Mini** tactile
sensors, one mounted on each finger. The sensors are simulated using a Phong
illumination model applied to per-pixel depth maps rendered by embedded MuJoCo
cameras. The simulation is copied from [gelsight_mini](https://gitlab.sdu.dk/pengu20/mj_sim/-/tree/f9336f6d4b8d44384ce7ebab3739813f66d34dfc/sensors/gelsight_mini) repo.

The new gripper class is `GripperPandaGelSightMini` and it is a drop-in
replacement for `GripperPanda` throughout the pipeline.

---

## Gripper Configuration

### Python class — `mgs/gripper/panda_gelsight_mini.py`

`GripperPandaGelSightMini` has the following key physical constants:

| Constant | Value | Meaning |
|---|---|---|
| `MAX_WIDTH` | 0.08 m | Maximum finger opening |
| `SENSOR_THICKNESS` | 0.01825 m | Total sensor depth (calibrated by `sensor_thickness.py`) |
| `ELASTOMER_THICKNESS` | 0.004 m | Deformable elastomer layer thickness |
| `MIN_WIDTH_CLAMP` | `2 × (SENSOR_THICKNESS − ELASTOMER_THICKNESS)` | Minimum achievable width (hard stop) |

The sensors are added to the MuJoCo XML as mesh geometries
(`gsmini_shell.stl`) positioned at `pos="0 0.005 0.04"` relative to each
finger body. A hard-stop box geom prevents the fingers from colliding through
each other when no object is present.

Note on the sensor thickness and placement:
- Physical Dimension: The tactile sensors have a physical width of 0.2825 m.
- To maximize the gripper's effective range, the sensors are intentionally offset backwards by 10 mm relative to the standard finger mount. Therefore the `SENSOR_THICKNESS` is referred to 0.01825 m.
- The calculation resulting in the final position of $y = 0.005\text{ m}$ in the configuration file stems from two main factors:
    - Mesh File Origin: The coordinate origin of the sensor's 3D mesh file does not sit flush with its rear casing. Instead, it is located 5 mm in front of the rear housing wall.
    - Coordinate Frame Inversion (Orientation): The coordinate frames of the finger and the sensor are inverted along the Y-axis. Specifically, the positive Y-axis ($+Y$) of the finger frame points in the exact opposite direction of the sensor's negative Y-axis ($-Y$).

Each finger also contains an embedded camera:

```xml
<camera name="tactile_cam_left"  pos="0 0.01 0.04" quat="0.707 -0.707 0 0" fovy="35" resolution="640 480"/>
<camera name="tactile_cam_right" pos="0 0.01 0.04" quat="0.707 -0.707 0 0" fovy="35" resolution="640 480"/>
```

These cameras look inward through the elastomer to capture the depth map that
the `GelSightMini` class converts into a tactile RGB image.

### Hydra config — `mgs/cli/config/gripper/panda_gelsight_mini.yaml`

```yaml
name: "PandaGripperGelsightMini"
id: "panda_gelsight_mini"
qpos: [0.04, 0.0]          # open
qpos_close: [0.0, -0.04]   # closed
grasp_sampler: "Antipodal"
eta: 7000
```

The gripper is selected automatically when `gripper: panda_gelsight_mini` is
set in any pipeline config.

### Width-to-joint mapping

`width_to_joints(width)` converts a desired gap between finger surfaces (in
metres) to actuator positions, compensating for the physical sensor thickness:

```
adjusted_width = width + 2 × SENSOR_THICKNESS
clamped_width  = clip(adjusted_width, MIN_WIDTH_CLAMP, MAX_WIDTH)
q1 = clamped_width / 2           # left finger (range 0 → 0.04)
q2 = -0.04 + clamped_width / 2  # right finger (range -0.04 → 0)
```

Passing `width = -1.0` guarantees that the fingers reach the hard stop
regardless of the object size — this is the standard "fully close" command used
throughout the pipeline.

### Coordinate transform

`base_to_contact_transform()` returns the SE(3) offset from the gripper base
body (`hand`) to the contact midpoint between the two sensor surfaces:

```python
pos  = np.array([0, 0, -0.09815])
quat = np.array([0.707106781, 0.0, 0.0, 0.707106781])  # wxyz
```

This value was determined experimentally using
`mgs/tactile_sensing/base_to_contact_transform.py` and is used in the scene
pipeline to convert saved object-frame grasp poses into world-frame gripper
base poses.

---

## Sensor Initialization

The `GelSightMini` class (`mgs/sensors/gelsight_mini/gelsight_mini.py`)
wraps an existing MuJoCo camera and converts its depth image into a simulated
tactile RGB image.

```python
from mgs.sensors.gelsight_mini.gelsight_mini import GelSightMini

class DummyArgs:
    save_dir = "./"
    cam_width = 640
    cam_height = 480

left_sensor  = GelSightMini(args=DummyArgs(), model=model, data=data, cam_name="tactile_cam_left")
right_sensor = GelSightMini(args=DummyArgs(), model=model, data=data, cam_name="tactile_cam_right")
```

The sensors require the MuJoCo model and data objects to already be
initialised. They read depth on demand — no persistent state is carried between
calls.

### Key sensor parameters

| Parameter | Value | Source |
|---|---|---|
| `_min_depth` | 0.02423 m | Camera-to-outer-surface distance |
| `_ELASTOMER_THICKNESS` | 0.004 m | Deformable layer depth |
| `_max_depth` | 0.02823 m | Outer contact boundary (`_min_depth + _ELASTOMER_THICKNESS`) |
| `_px2m_ratio` | 5.4348 × 10⁻⁵ m/px | Physical pixel scale |

### Getting tactile output

```python
# After mujoco.mj_step(model, data):
rgb_image   = left_sensor.tactile_image        # np.ndarray, shape (480, 640, 3), dtype uint8
depth_image = left_sensor.tactile_depth_image  # np.ndarray, shape (480, 640), float64
```

`tactile_image` applies Phong illumination with four coloured light sources
(white, blue, red, green) over an elastic deformation of the depth map,
producing an image that resembles output from a real GelSight Mini sensor.

---

## Output Data Structure

`render_scene_with_tactile_feedback.py` saves two `.npz` files per grasp:

```
outputs/02_scene/<gripper>/<scene_hash>/tactile_feedback/<object_stem>/
├── grasp_00000.npz
├── lift_00000.npz
├── grasp_00001.npz
├── lift_00001.npz
└── ...
```

### `grasp_<idx>.npz`

Recorded during the closing phase (gripper moves from open to contact).

| Key | Shape / Type | Description |
|---|---|---|
| `left_cam` | `(N, 480, 640, 3)` uint8 | Tactile RGB frames, left finger |
| `right_cam` | `(N, 480, 640, 3)` uint8 | Tactile RGB frames, right finger |
| `contact_frame_idx` | int | Frame index when first contact was detected (−1 if none) |
| `contact_step` | int | Simulation step of first contact |
| `fully_closed_frame_idx` | int | Frame index of the last closing frame |
| `grasp_label` | str | `"stable"` or `"failed"` |
| `pose_world` | `(4, 4)` float | Gripper contact pose in world frame (SE3 matrix) |
| `joints` | `(2,)` float | Target actuator positions at closing time |

### `lift_<idx>.npz`

Recorded during the lifting phase (gripper moves upward by `lift_dist`).

| Key | Shape / Type | Description |
|---|---|---|
| `left_cam` | `(M, 480, 640, 3)` uint8 | Tactile RGB frames, left finger |
| `right_cam` | `(M, 480, 640, 3)` uint8 | Tactile RGB frames, right finger |
| `grasp_success` | bool | `True` if contact was maintained throughout the lift |
| `contact_lost_step` | int | Simulation step when contact was lost (−1 if never lost) |
| `grasp_label` | str | `"stable"` or `"failed"` |
| `pose_world` | `(4, 4)` float | Gripper contact pose in world frame |
| `joints` | `(2,)` float | Actuator positions used during lift |

---

## How to Run the Pipeline

### Prerequisites

```bash
conda activate mj-grasp-sim
export MGS_INPUT_DIR=/path/to/your/in
export MGS_OUTPUT_DIR=/path/to/your/out
```

### Step 1 — Generate per-object grasps

Samples antipodal grasps for each object in `fast_eta_objects.txt`, filters
for collision-free and dynamically stable grasps, and saves them to
`outputs/01_grasp/`.

```bash
python -m mgs.cli.gen_gripper_object_grasps
```

Config: `mgs/cli/config/gen_gripper_object_grasps.yaml`
— uses `gripper: panda_gelsight_mini` by default.

### Step 2 — Generate clutter scenes

Drops multiple objects onto a simulated table, translates the per-object grasps
from Step 1 into world coordinates, and labels them as stable / collision /
failed. Saves scenes to `outputs/02_scene/`.

```bash
python -m mgs.cli.gen_scene
```

Config: `mgs/cli/config/gen_scene.yaml`

### Step 3 — Render tactile feedback

Replays every saved grasp in every scene, records tactile camera streams for
both the closing and lifting phases, and writes them alongside the scene files.

```bash
python -m mgs.cli.render_scene_with_tactile_feedback
```

Config: `mgs/cli/config/render_scene_with_tactile_feedback.yaml`

Key config parameters:

| Parameter | Default | Description |
|---|---|---|
| `save_tactile` | `true` | Whether to save `.npz` output |
| `visualize` | `false` | Enable live OpenCV + MuJoCo viewer |
| `steps_close` | 150 | Simulation steps for the closing phase |
| `steps_lift` | 500 | Simulation steps for the lifting phase |
| `lift_dist` | 0.3 m | Vertical distance lifted |
| `render_interval` | 1 | Capture a frame every N simulation steps |

Keyboard controls when `visualize: true`:

| Key | Action |
|---|---|
| `q` | Quit entirely |
| `n` | Skip to next scene |
| `l` | Skip remaining grasps of the current label type |
| `s` | Skip current grasp |


---

## Utility Scripts — `mgs/tactile_sensing/`

These scripts are not part of the automated pipeline but are useful for
calibration and data inspection.

### `sensor_thickness.py`

Measures the physical thickness of the GelSight Mini sensor mesh by slowly
closing the gripper in a gravity-free simulation until the two sensor geoms
collide. The measured value is used to set `SENSOR_THICKNESS` in
`GripperPandaGelSightMini`.

```bash
python -m mgs.tactile_sensing.sensor_thickness
```

### `play_tactile_stream.py`

Plays back saved tactile `.npz` files as side-by-side video (left and right
finger). Overlays contact annotations and success/failure labels. Processes all
grasp–lift pairs found under `$MGS_OUTPUT_DIR/02_scene/`.

```bash
python -m mgs.tactile_sensing.play_tactile_stream
```

Controls: `space` pause/advance · `n` next file · `q` quit.

### `base_to_contact_transform.py`

Calibration script that determines the SE(3) offset between the gripper base
body and the contact midpoint. Closes the gripper onto a small reference box
and computes the local transform. The result is hard-coded into
`GripperPandaGelSightMini.base_to_contact_transform()`.

```bash
python -m mgs.tactile_sensing.base_to_contact_transform
```

---

## Asset Files

| Path | Description |
|---|---|
| `asset/sensors/gelsight_mini/gsmini_shell.stl` | Collision and visual mesh for the sensor housing |
| `mgs/sensors/gelsight_mini/assets/background_gelsight2017.jpg` | Background texture for tactile image synthesis |

---

## Acknowledgements

The `GelSightMini` simulation is based on the method published by
[Daniel Fernandes Gomes](https://github.com/danfergo/gelsight_simulation).
The Panda gripper XML is derived from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0 licence, see `3rd-party-licenses.txt`).
