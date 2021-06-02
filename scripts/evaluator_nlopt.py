import itertools

from evolution.base import FitnessStrategy
from evolution.base.base_geometry import BaseGeometry
from evolution.camera import CameraGenomeParameters
from loguru import logger

from cameras.cameras import Amount, wiggle_camera
from optimizer.nlopt_optimizer import NloptAlgorithms, NloptOptimizer
from scripts.evaluator_base import Evaluator
from utils.noise_utils import NoiseStrategy
from utils.persistence_utils import NloptResultWriter


class NloptEvaluator(Evaluator):
    def __init__(
            self,
            image_shape: tuple[int, int],
            genome_parameters: CameraGenomeParameters,
            output_file: str,
            append_mode: bool = True,
    ):
        super().__init__(image_shape, genome_parameters)

        self._evolution_writer: ResultWriter = NloptResultWriter(
            output_file, append_mode=append_mode
        )

    def evaluate(
            self,
            fitting_geometry: BaseGeometry,
            amounts: list[Amount],
            fitness_strategies: list[FitnessStrategy],
            nlopt_algorithms: list[NloptAlgorithms],
            noise_strategies: list[NoiseStrategy],
            runs_per_bundle: int = 32,
            headless: bool = True,
    ):

        strategies = [amounts, fitness_strategies, nlopt_algorithms, noise_strategies]
        for (amount, fitness_strategy, nlopt_algorithm, noise_strategy) in itertools.product(
                *strategies
        ):

            for _ in range(runs_per_bundle):
                start_camera = wiggle_camera(
                    self._target_camera, amount, self._wiggle_indices, self._n_wiggles
                )
                edge_image = self._construct_edge_image(
                    self._image_shape,
                    self._camera_translator,
                    self._target_genome,
                    fitting_geometry,
                    noise_strategy,
                )

                nlopt_optimizer = NloptOptimizer(
                    fitness_strategy,
                    edge_image,
                    start_camera.dna,
                    fitting_geometry,
                    nlopt_algorithm,
                    headless,
                )
                nlopt_result = nlopt_optimizer.optimize()
                logger.info(
                    "NLOPT {} took {}", nlopt_result.result_code, nlopt_result.optimize_duration
                )
