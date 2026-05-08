# Copyright (c) 2025 Robert Bosch GmbH
# Author: Roman Freiberg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import pickle
from typing import List

import numpy as np
from omegaconf import DictConfig

from mgs.obj.base import CollisionMeshObject
from mgs.obj.cube import ObjectCube
from mgs.obj.gso import ObjectGSO
from mgs.obj.ycb import ObjectYCB
from mgs.obj.angle_grinder import ObjectAngleGrinder
from mgs.util.const import GIT_PATH
from mgs.util.file import generate_unique_hash
from mgs.util.geo.transforms import SE3Pose


def get_object(id: int) -> CollisionMeshObject:
    ycb_obj_ids = [o for o in ObjectYCB.all_object_ids() if o == id]
    gso_obj_ids = [o for o in ObjectGSO.all_object_ids() if o == id]
    angle_grinder_ids = ["angle_grinder"] if "angle_grinder" == id else []   
    assert len(ycb_obj_ids) + len(gso_obj_ids) + len(angle_grinder_ids) == 1
    if len(gso_obj_ids) == 1:
        hash_name = generate_unique_hash()
        return ObjectGSO(
            SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
            object_id=gso_obj_ids[0],
            name=hash_name,
        )
    if len(ycb_obj_ids) == 1:
        hash_name = generate_unique_hash()
        return ObjectYCB(
            SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
            object_id=ycb_obj_ids[0],
            name=hash_name,
        )
    if len(angle_grinder_ids) == 1:
        hash_name = generate_unique_hash()
        return ObjectAngleGrinder(
            SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
            name=hash_name,
        )
    raise ValueError("Object not found")


def get_objects(cfg: DictConfig) -> List[CollisionMeshObject]:
    object_list = []

    if cfg.name == "SingleObject":
        ycb_obj_ids = [o for o in ObjectYCB.all_object_ids() if o == cfg.id]
        gso_obj_ids = [o for o in ObjectGSO.all_object_ids() if o == cfg.id]
        angle_grinder_ids = ["angle_grinder"] if cfg.id == "angle_grinder" else []  
        assert len(ycb_obj_ids) + len(gso_obj_ids) + len(angle_grinder_ids) == 1
        if len(angle_grinder_ids) == 1:
            hash_name = generate_unique_hash()
            object_list.append(
                ObjectAngleGrinder(
                    SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
                    name=hash_name,
                )
            )
        if len(gso_obj_ids) == 1:
            hash_name = generate_unique_hash()
            object_list.append(
                ObjectGSO(
                    SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
                    object_id=gso_obj_ids[0],
                    name=hash_name,
                )
            )
        if len(ycb_obj_ids) == 1:
            hash_name = generate_unique_hash()
            object_list.append(
                ObjectYCB(
                    SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
                    object_id=ycb_obj_ids[0],
                    name=hash_name,
                )
            )
    elif cfg.name == "YCB":
        ycb_obj_ids = ObjectYCB.all_object_ids()
        ycb_obj_ids.sort()
        for obj in ycb_obj_ids:
            hash_name = generate_unique_hash()
            object_list.append(
                ObjectYCB(
                    SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
                    object_id=obj,
                    name=hash_name,
                )
            )
    elif cfg.name == "GSO":
        for obj in ObjectGSO.all_object_ids():
            hash_name = generate_unique_hash()
            object_list.append(
                ObjectGSO(
                    SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
                    object_id=obj,
                    name=hash_name,
                )
            )
    elif cfg.name == "Full_Dataset":
        gso_object_ids = ObjectGSO.all_object_ids()
        gso_object_ids.sort()
        for obj in gso_object_ids:
            hash_name = generate_unique_hash()
            object_list.append(
                ObjectGSO(
                    SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
                    object_id=obj,
                    name=hash_name,
                )
            )
        ycb_obj_ids = ObjectYCB.all_object_ids()
        ycb_obj_ids.sort()
        for obj in ycb_obj_ids:
            hash_name = generate_unique_hash()
            object_list.append(
                ObjectYCB(
                    SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
                    object_id=obj,
                    name=hash_name,
                )
            )
    elif cfg.name == "Full_Data_Subset":
        import random

        num_objects_min = cfg.num_objects_min
        num_objects_max = cfg.num_objects_max
        # choose a random number for the number of objects
        num_objects = random.randint(num_objects_min, num_objects_max)

        graspable_object_set_file = os.path.join(
            GIT_PATH, "mgs", "cli", "stats", "graspable_object_set.pickle"
        )
        # Unpickle the set (deserialization)
        with open(graspable_object_set_file, "rb") as f:
            graspable_object_set = pickle.load(f)

        # choose num_objects (with repetition) from the list of all object ids
        ycb_obj_ids = [
            ("ycb", i) for i in ObjectYCB.all_object_ids() if i in graspable_object_set
        ]
        gso_obj_ids = [
            ("gso", i) for i in ObjectGSO.all_object_ids() if i in graspable_object_set
        ]
        obj_ids = ycb_obj_ids + gso_obj_ids
        chosen_obj_ids = random.choices(obj_ids, k=num_objects)

        x, y = -8.5, -8
        for i, tagged_obj in enumerate(chosen_obj_ids):
            tag, obj = tagged_obj
            hash_name = generate_unique_hash()

            if i % 10 == 0:
                x += 0.5
                y = -8
            else:
                y += 0.5
            if tag == "ycb":
                object_list.append(
                    ObjectYCB(
                        SE3Pose(
                            np.array([float(x), float(y), 0]),
                            np.array([1, 0, 0, 0]),
                            type="wxyz",
                        ),
                        object_id=obj,
                        name=hash_name,
                    )
                )
            elif tag == "gso":
                object_list.append(
                    ObjectGSO(
                        SE3Pose(
                            np.array([float(x), float(y), 0]),
                            np.array([1, 0, 0, 0]),
                            type="wxyz",
                        ),
                        object_id=obj,
                        name=hash_name,
                    )
                )
    elif cfg.name == "ObjectList":
        import random

        # Ensure cfg.ids is treated as a list for exact matching
        if isinstance(cfg.ids, str):
            ids_list = [cfg.ids]
        else:
            ids_list = list(cfg.ids)

        ycb_obj_ids = [("ycb", o) for o in ObjectYCB.all_object_ids() if o in ids_list]
        gso_obj_ids = [("gso", o) for o in ObjectGSO.all_object_ids() if o in ids_list]
        angle_grinder_ids = [("angle_grinder", "angle_grinder")]  if "angle_grinder" in ids_list else [] 

        obj_ids = ycb_obj_ids + gso_obj_ids + angle_grinder_ids

        # If parent config specifies num_objects, respect that limit
        if hasattr(cfg, 'num_objects') and cfg.num_objects is not None:
            if len(obj_ids) > cfg.num_objects:
                raise ValueError(f"more than {cfg.num_objects} objecs selected.")

        x, y = -8.5, -8
        for i, tagged_obj in enumerate(obj_ids):
            tag, obj = tagged_obj
            hash_name = generate_unique_hash()

            if i % 10 == 0:
                x += 0.5
                y = -8
            else:
                y += 0.5
            if tag == "ycb":
                object_list.append(
                    ObjectYCB(
                        SE3Pose(
                            np.array([float(x), float(y), 0]),
                            np.array([1, 0, 0, 0]),
                            type="wxyz",
                        ),
                        object_id=obj,
                        name=hash_name,
                    )
                )
            elif tag == "gso":
                object_list.append(
                    ObjectGSO(
                        SE3Pose(
                            np.array([float(x), float(y), 0]),
                            np.array([1, 0, 0, 0]),
                            type="wxyz",
                        ),
                        object_id=obj,
                        name=hash_name,
                    )
                )
            elif tag == "angle_grinder":
                object_list.append(
                    ObjectAngleGrinder(
                        SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
                        name=hash_name,
                    )
                )
    elif cfg.name == "Cube":
        hash_name = generate_unique_hash()
        object_list.append(
            ObjectCube(
                SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz"),
                name=hash_name,
                size=0.02,
            )
        )
    elif cfg.name == "Fast_Data_Subset":
        import random

        num_objects = cfg.num_objects
        fast_object_file = os.path.join(
            GIT_PATH, "asset", "mj-objects", "obj_unsymmetric_train.txt"
        )
        with open(fast_object_file, "r") as file:
            fast_objects = file.read().splitlines()

        # choose num_objects (with repetition) from the list of all object ids
        ycb_obj_ids = [
            ("ycb", i) for i in ObjectYCB.all_object_ids() if i in fast_objects
        ]
        gso_obj_ids = [
            ("gso", i) for i in ObjectGSO.all_object_ids() if i in fast_objects
        ]
        obj_ids = ycb_obj_ids + gso_obj_ids
        chosen_obj_ids = random.sample(obj_ids, k=num_objects)

        x, y = -8.5, -8
        for i, tagged_obj in enumerate(chosen_obj_ids):
            tag, obj = tagged_obj
            hash_name = generate_unique_hash()

            if i % 10 == 0:
                x += 0.5
                y = -8
            else:
                y += 0.5
            if tag == "ycb":
                object_list.append(
                    ObjectYCB(
                        SE3Pose(
                            np.array([float(x), float(y), 0]),
                            np.array([1, 0, 0, 0]),
                            type="wxyz",
                        ),
                        object_id=obj,
                        name=hash_name,
                    )
                )
            elif tag == "gso":
                object_list.append(
                    ObjectGSO(
                        SE3Pose(
                            np.array([float(x), float(y), 0]),
                            np.array([1, 0, 0, 0]),
                            type="wxyz",
                        ),
                        object_id=obj,
                        name=hash_name,
                    )
                )
    return object_list
