# Copyright (c) 2025 Robert Bosch GmbH
# Author: Roman Freiberg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License...

import os
import xml.etree.ElementTree as Et
from typing import Any, Dict, Tuple
from mgs.util.const import ASSET_PATH

from mgs.obj.base import CollisionMeshObject
from mgs.util.geo.transforms import SE3Pose


class ObjectAngleGrinder(CollisionMeshObject):
    """
    Simple MuJoCo object from a single OBJ file.

    Supports:
    - Visual mesh
    - Collision mesh (same geometry)
    - Optional MTL parsing (Kd color only)
    """

    def __init__(
        self,
        pose: SE3Pose,
        name: str = None,
        mass: float = 1.0,
    ):
        
        pose_vec = pose.to_vec(layout="pq", type="wxyz")
        pos, quat = pose_vec[:3], pose_vec[3:]

        self.pos = pos
        self.quat = quat

        self.asset_dir = os.path.join(ASSET_PATH, "mj-objects/angle_grinder")
        self.obj_path = os.path.join(ASSET_PATH, "mj-objects/angle_grinder/angle_grinder")
        self.object_id = "angle_grinder"
        self.name = self.object_id if name is None else name
        self.file_name = "{}.xml".format(self.object_id)

        self.mass = mass

    @property
    def obj_file_path(self):
        return os.path.join(self.obj_path, "textured.obj")

    def _parse_mtl_for_color(self):
        """
        Extract diffuse color (Kd) from MTL file.
        Returns (r, g, b) or None.
        """
        mtl_path = os.path.join(self.obj_path, "textured.mtl")

        if not os.path.isfile(mtl_path):
            return None

        with open(mtl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Kd"):
                    parts = line.split()
                    if len(parts) == 4:
                        try:
                            return tuple(map(float, parts[1:]))
                        except ValueError:
                            return None
        return None

    def to_xml(self) -> Tuple[str, Dict[str, Any]]:
        xml_string = self.generate_xml()
        return (
            '<include file="{}_{}" />'.format(self.name, self.file_name),
            {"{}_{}".format(self.name, self.file_name): xml_string},
        )

    def generate_xml(self) -> bytes:
        root = Et.Element("mujoco", attrib={"model": self.name})

        # Assets and Worldbody
        assets = Et.SubElement(root, "asset")
        worldbody = Et.SubElement(root, "worldbody")

        body_attributes = {
            "name": self.name,
            "pos": " ".join(map(str, self.pos)),
            "quat": " ".join(map(str, self.quat)),
        }
        body = Et.SubElement(worldbody, "body", attrib=body_attributes)

        # --- Mesh asset ---
        mesh_attributes = {
            "name": f"{self.name}_mesh",
            "file": self.obj_file_path,
        }
        Et.SubElement(assets, "mesh", attrib=mesh_attributes)

        # --- Material (from MTL color if available) ---
        color = self._parse_mtl_for_color()
        material_name = None

        if color is not None:
            rgba = f"{color[0]} {color[1]} {color[2]} 1"
            material_attributes = {
                "name": f"{self.name}_mat",
                "rgba": rgba,
                "specular": "0.5",
                "shininess": "0.5",
            }
            Et.SubElement(assets, "material", attrib=material_attributes)
            material_name = material_attributes["name"]

        # --- Visual geom ---
        visual_geom_attributes = {
            "mesh": mesh_attributes["name"],
            "type": "mesh",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
        }

        if material_name is not None:
            visual_geom_attributes["material"] = material_name
        else:
            # fallback gray color
            visual_geom_attributes["rgba"] = "0.7 0.7 0.7 1"

        Et.SubElement(body, "geom", attrib=visual_geom_attributes)

        # --- Collision geom ---
        collision_geom_attributes = {
            "mesh": mesh_attributes["name"],
            "type": "mesh",
            "mass": str(self.mass),
            "group": "3",
            "contype": "1",
            "conaffinity": "1",
            "condim": "4",
            "friction": "1.0 0.3 0.1",
            "solimp": "0.998 0.998 0.001",
            "solref": "0.001 1",
        }
        Et.SubElement(body, "geom", attrib=collision_geom_attributes)

        # --- Free joint ---
        joint_attributes = {
            "name": f"{self.name}:joint",
            "type": "free",
            "damping": "0.0001",
        }
        Et.SubElement(body, "joint", attrib=joint_attributes)

        return Et.tostring(root)