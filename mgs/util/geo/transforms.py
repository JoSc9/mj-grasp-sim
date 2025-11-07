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

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing_extensions import Self

from .convert import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from .operations import quaternion_apply, quaternion_invert


@dataclass
class SE3Pose:
    pos: np.ndarray
    quat: np.ndarray  # as quat
    type: str  # only options are "wxyz" or "xyzw"

    def __post_init__(self):
        assert self.pos.shape[-1] == 3
        assert self.quat.shape[-1] == 4
        assert self.type == "wxyz" or self.type == "xyzw"

        # if numpy array, convert to torch
        self.pos = self.pos.astype(np.float32)
        self.quat = self.quat.astype(np.float32)

        norms = np.sum(self.quat**2, axis=-1, keepdims=True)
        ones = np.ones_like(self.quat[..., [0]])
        assert np.all(np.isclose(norms, ones, rtol=1e-4))

    def to_vec(self, layout="pq", type=None) -> np.ndarray:
        quat: np.ndarray = np.copy(self.quat)
        if type is not None and type != self.type:
            if type == "wxyz":
                quat = quat_xyzw_to_wxyz(quat)
            elif type == "xyzw":
                quat = quat_wxyz_to_xyzw(quat)
            else:
                raise ValueError

        out = np.array([])
        if layout == "pq":
            out = np.concatenate([self.pos, quat], axis=-1)
        elif layout == "qp":
            out = np.concatenate([quat, self.pos], axis=-1)
        return out

    @classmethod
    def from_vec(cls, vec: np.ndarray, type: str = "wxyz", layout: str = "pq") -> Self:
        assert vec.shape[-1] == 7

        if layout == "pq":
            pos = vec[..., -7:-4]
            quat = vec[..., -4:]
        elif layout == "qp":
            pos = vec[..., 4:7]
            quat = vec[..., 0:4]
        else:
            raise ValueError

        return cls(pos, quat, type)

    @classmethod
    def from_mat(cls, mat: np.ndarray, type: str = "wxyz") -> Self:
        assert mat.shape[-2:] == (4, 4)
        quat = R.from_matrix(mat[..., :3, :3]).as_quat(canonical=False)
        if type == "wxyz":
            quat = quat_xyzw_to_wxyz(quat)
        else:
            raise ValueError

        return cls(mat[..., :3, 3], quat, type)

    def __getitem__(self, idx: int) -> Self:
        return self.__class__(self.pos[idx], self.quat[idx], self.type)

    def __matmul__(self, other: Self) -> Self:
        self_mat = self.to_mat()
        other_mat = other.to_mat()
        res_mat = np.einsum("...ij,...jk->...ik", self_mat, other_mat)
        return self.__class__.from_mat(res_mat, type=self.type)

    def __len__(self) -> int:
        return len(self.pos)

    def inverse(self):
        inv_quat = quaternion_invert(self.quat, type=self.type)
        inv_pos = -quaternion_apply(inv_quat, self.pos, type=self.type)
        self.pos = inv_pos
        self.quat = inv_quat
        return SE3Pose(inv_pos, inv_quat, self.type)

    def to_mat(self) -> np.ndarray:
        if self.type == "wxyz":
            quat_scipy = quat_wxyz_to_xyzw(self.quat)
        else:
            quat_scipy = self.quat

        mat = np.zeros((*self.quat.shape[:-1], 4, 4), dtype=np.float32)
        quat_scipy_cpu = np.copy(quat_scipy)
        mat_rot = R.from_quat(quat_scipy_cpu).as_matrix()
        mat[..., :3, :3] = mat_rot
        mat[..., :3, 3] = self.pos
        mat[..., 3, 3] = 1.0
        return mat

    @classmethod
    def randn_se3(cls, num) -> Self:
        scipy_quat = R.random(num).as_quat(canonical=False)
        quat = quat_xyzw_to_wxyz(scipy_quat)
        pos = np.random.randn(num, 3)
        return cls(pos, quat, "wxyz")
    
    @classmethod
    def randn_se3_perturb(cls, num, sigma_rot=0.05, sigma_trans=0.01) -> Self:

        w = sigma_rot * np.random.rand(num, 3)

        quat = R.from_rotvec(w).as_quat()
        quat = quat_xyzw_to_wxyz(quat)

        pos = sigma_trans * np.random.randn(num, 3)
        return cls(pos, quat, "wxyz")

