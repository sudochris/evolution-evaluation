from typing import Tuple

import numpy as np
from evolution.base.base_geometry import BaseGeometry
from evolution.camera import (
    CameraGenomeParameters,
    render_geometry_with_camera,
)

from cameras.cameras import mean_squash_camera, Camera
from utils.color_utils import Color
from utils.noise_utils import NoiseStrategy, add_noise_to_image
from utils.transform_utils import split_dna


class Evaluator:
    def __init__(self, image_shape: Tuple[int, int], genome_parameters: CameraGenomeParameters):
        super().__init__()
        self._image_shape = image_shape

        self._genome_parameters = genome_parameters
        self._target_camera = mean_squash_camera(image_shape)

        self._wiggle_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self._n_wiggles = 7

    def _construct_edge_image(
            self,
            image_shape: Tuple[int, int],
            camera: Camera,
            fitting_geometry: BaseGeometry,
            noise_strategy: NoiseStrategy,
    ) -> np.array:
        edge_image = np.zeros(image_shape, dtype=np.uint8)

        cm, t, r, d = split_dna(camera.dna)
        render_geometry_with_camera(edge_image, fitting_geometry, cm, t, r, d, Color.WHITE)

        noise_image = noise_strategy.generate_noise(edge_image)
        return add_noise_to_image(edge_image, noise_image)
