import itertools
from typing import List, Tuple

from evolution.base import FitnessStrategy
from evolution.base.base_geometry import BaseGeometry, DenseGeometry, PlaneGeometry
from evolution.camera import CameraGenomeParameters
from loguru import logger

from cameras.cameras import Amount, wiggle_camera, Camera
from optimizer.nlopt_optimizer import NloptOptimizer, NloptAlgorithm
from scripts.evaluator_base import Evaluator
from utils.error_utils import reprojection_error_multiple_geometries
from utils.noise_utils import NoiseStrategy
from utils.persistence_utils import NloptResultWriter


class NloptEvaluator(Evaluator):
    def __init__(
            self,
            image_shape: Tuple[int, int],
            genome_parameters: CameraGenomeParameters,
            output_file: str,
            append_mode: bool = True,
    ):
        super().__init__(image_shape, genome_parameters)

        self._nlopt_writer: NloptResultWriter = NloptResultWriter(
            output_file, append_mode=append_mode
        )

    def evaluate(
            self,
            fitting_geometry: BaseGeometry,
            amounts: List[Amount],
            fitness_strategies: List[FitnessStrategy],
            nlopt_algorithms: List[NloptAlgorithm],
            noise_strategies: List[NoiseStrategy],
            runs_per_bundle: int = 32,
            headless: bool = True,
    ):

        geometry_dense = DenseGeometry(fitting_geometry, 16)
        geometry_y0 = PlaneGeometry(fitting_geometry, 0, 16)
        evaluation_geometries = [fitting_geometry, geometry_dense, geometry_y0]
        strategies = [amounts, fitness_strategies, nlopt_algorithms, noise_strategies]

        for (amount, fitness_strategy, nlopt_algorithm, noise_strategy) in itertools.product(
                *strategies
        ):
            algorithm_name = nlopt_algorithm.display_name

            n_performed_experiments = self._nlopt_writer.has(
                amount,
                algorithm_name,
                noise_strategy
            )

            if n_performed_experiments > 0:
                printable_experiment_string = "{} {} {}".format(amount.name, algorithm_name,
                                                                noise_strategy.printable_identifier())
                delta = runs_per_bundle - n_performed_experiments
                logger.info(
                    f"Found {n_performed_experiments} experiments {delta} missing [{printable_experiment_string}]"
                )

            for _ in range(n_performed_experiments, runs_per_bundle):
                start_camera = wiggle_camera(
                    self._target_camera, amount, self._wiggle_indices, self._n_wiggles
                )
                edge_image = self._construct_edge_image(
                    self._image_shape,
                    self._target_camera,
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
                    "{} took {} [{}]", algorithm_name, nlopt_result.optimize_duration, nlopt_result.result_code
                )

                result_camera = Camera.from_dna(nlopt_result.best_result)

                reprojection_errors = reprojection_error_multiple_geometries(
                    start_camera, result_camera.dna, evaluation_geometries
                )

                fitting_geometry_result = reprojection_errors[0]
                dense_geometry_result = reprojection_errors[1]
                y0_geometry_result = reprojection_errors[2]

                best_fitness = nlopt_result.best_fitness
                duration_in_s = nlopt_result.optimize_duration

                self._nlopt_writer.save_experiment(
                    algorithm_name,
                    amount,
                    start_camera,
                    self._target_camera,
                    result_camera,
                    noise_strategy,
                    fitting_geometry_result,
                    dense_geometry_result,
                    y0_geometry_result,
                    best_fitness,
                    duration_in_s
                )
