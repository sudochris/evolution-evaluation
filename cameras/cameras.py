import random
from typing import Tuple

import numpy as np


class Amount:
    def __init__(self, values: np.array, name: str):
        """Represents the range limit for wiggling parameters

        Basically this is a named wrapper for a 15 element vector

        Args:
            values (np.array): $\pm$ limits per parameters
            name (str): A representative name
        """
        self._values = values
        self._name = name

    @property
    def values(self):
        return self._values

    @property
    def name(self):
        return self._name

    # region [Region0] Static methods for "near", "medium" and "far"
    @staticmethod
    def near() -> "Amount":
        return Amount(
            [10, 10, 10, 10, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.025, 0.01, 0.5], "Near"
        )

    @staticmethod
    def medium() -> "Amount":
        return Amount(
            [50, 50, 50, 50, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.25, 0.25, 0.1, 0.05, 1.5], "Medium"
        )

    @staticmethod
    def far() -> "Amount":
        return Amount(
            [100, 100, 100, 100, 1.0, 2.0, 1.0, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2, 0.1, 3.0], "Far"
        )

    # endregion


class Camera:
    def __init__(
            self,
            fu: float = 0.0,
            fv: float = 0.0,
            cx: float = 0.0,
            cy: float = 0.0,
            tx: float = 0.0,
            ty: float = 0.0,
            tz: float = 0.0,
            rx: float = 0.0,
            ry: float = 0.0,
            rz: float = 0.0,
            d0: float = 0.0,
            d1: float = 0.0,
            d2: float = 0.0,
            d3: float = 0.0,
            d4: float = 0.0,
    ):
        self._fu, self._fv = fu, fv
        self._cx, self._cy = cx, cy
        self._tx, self._ty, self._tz = tx, ty, tz
        self._rx, self._ry, self._rz = rx, ry, rz
        self._d0, self._d1, self._d2, self._d3, self._d4 = d0, d1, d2, d3, d4

        self._dna = np.array([fu, fv, cx, cy, tx, ty, tz, rx, ry, rz, d0, d1, d2, d3, d4])

    @staticmethod
    def from_dna(dna: np.array) -> "Camera":
        return Camera(*dna)

    @property
    def dna(self):
        return self._dna


def mean_squash_camera(image_shape: Tuple[int, int]) -> Camera:
    fuv = max(image_shape) - 100
    h, w = image_shape

    cx, cy = w // 2, h // 2
    tx, ty, tz = 0.0, 2.35, 8.4
    rx, ry, rz = 0.29, 0.0, 0.0
    return Camera(fuv, fuv, cx, cy, tx, ty, tz, rx, ry, rz)


def wiggle_camera(
        camera: Camera, amount: Amount, allowed_gene_indices: list, n_genes: int
) -> Camera:
    """Selects a number of allowed genes and changes them according the specified amount

    Args:
        camera (Camera): The initial camera
        amount (Amount): The amount to wiggle
        allowed_gene_indices (list): Allowed parameters (see gene encoding)
        n_genes (int): Number of parameters to select

    Returns:
        Camera: The resulting camera
    """
    new_dna = camera.dna.copy()  # 1. Copy initial camera dna
    idx = random.sample(allowed_gene_indices, n_genes)  # 2. Sample allowed indices
    lookup = np.zeros_like(new_dna)  # 3. Allocate an array for
    lookup[idx] = np.random.choice([-1, 1], n_genes)  # 4. Select positive or negative values
    new_dna += amount.values * lookup  # 4. Add variation to new dna
    return Camera(*new_dna)  # 5. Construct the new camera from the new dna and return
