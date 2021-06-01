import numpy as np
from evolution.base import BaseGenome
from evolution.base.base_geometry import BaseGeometry
from evolution.camera import render_geometry_with_camera
from evolution.camera.camera_translator import CameraTranslator
from utils.color_utils import Color
from utils.noise_utils import NoiseStrategy, add_noise_to_image


class Evaluator:
    def _construct_edge_image(
        self,
        image_shape: tuple[int, int],
        translator: CameraTranslator,
        genome: BaseGenome,
        fitting_geometry: BaseGeometry,
        noise_strategy: NoiseStrategy,
    ) -> np.array:

        edge_image = np.zeros(image_shape, dtype=np.uint8)

        cm, t, r, d = translator.translate_genome(genome)
        render_geometry_with_camera(edge_image, fitting_geometry, cm, t, r, d, Color.WHITE)

        noise_image = noise_strategy.generate_noise(edge_image)
        return add_noise_to_image(edge_image, noise_image)
