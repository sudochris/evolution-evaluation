import itertools

import cv2 as cv
import numpy as np
from cameras.cameras import Amount, Camera, mean_squash_camera, wiggle_camera
from evolution.base import (
    BaseGenome,
    BaseGeometry,
    CrossoverStrategy,
    FitnessStrategy,
    MutationStrategy,
    PopulateStrategy,
    SelectionStrategy,
    TerminationStrategy,
)
from evolution.base.base_geometry import DenseGeometry, PlaneGeometry
from evolution.camera import (
    CameraGenomeFactory,
    CameraGenomeParameters,
    CameraTranslator,
    render_geometry_with_camera,
)
from evolution.strategies import StrategyBundle
from loguru import logger
from optimizer.evolution_optimizer import EvolutionOptimizer
from utils.color_utils import Color
from utils.error_utils import reprojection_error_multiple_geometries
from utils.noise_utils import NoiseStrategy, add_noise_to_image
from utils.persistence_utils import EvolutionResultWriter, ResultWriter

from scripts.evaluator_base import Evaluator


class EvolutionEvaluator(Evaluator):
    def __init__(
        self,
        image_shape: tuple[int, int],
        genome_parameters: CameraGenomeParameters,
        output_file: str,
        append_mode: bool = True,
    ):
        super().__init__(image_shape, genome_parameters)

        self._evolution_writer: ResultWriter = EvolutionResultWriter(
            output_file, append_mode=append_mode
        )

    def evaluate(
        self,
        fitting_geometry: BaseGeometry,
        amounts: list[Amount],
        population_strategies: list[PopulateStrategy],
        fitness_strategies: list[FitnessStrategy],
        selection_strategies: list[SelectionStrategy],
        crossover_strategies: list[CrossoverStrategy],
        mutation_stategies: list[MutationStrategy],
        termination_strategies: list[TerminationStrategy],
        noise_strategies: list[NoiseStrategy],
        runs_per_bundle=32,
        headless=True,
    ):
        geometry_dense = DenseGeometry(fitting_geometry, 16)
        geometry_y0 = PlaneGeometry(fitting_geometry, 0, 16)
        evaluation_geometries = [fitting_geometry, geometry_dense, geometry_y0]

        strategies = [
            amounts,
            population_strategies,
            fitness_strategies,
            selection_strategies,
            crossover_strategies,
            mutation_stategies,
            termination_strategies,
            noise_strategies,
        ]
        for (
            amount,
            population_strategy,
            fitness_strategy,
            selection_strategy,
            crossover_strategy,
            mutation_strategy,
            termination_strategy,
            noise_strategy,
        ) in itertools.product(*strategies):

            n_performed_experiments = self._evolution_writer.has(
                amount,
                population_strategy(np.zeros(15)),
                fitness_strategy,
                selection_strategy,
                crossover_strategy,
                mutation_strategy,
                noise_strategy,
            )
            if n_performed_experiments > 0:
                logger.info(
                    f"Found {n_performed_experiments} experiments {runs_per_bundle-n_performed_experiments} missing."
                )
            for _ in range(n_performed_experiments, runs_per_bundle):
                start_camera = wiggle_camera(
                    self._target_camera, amount, self._wiggle_indices, self._n_wiggles
                )

                strategy_bundle = StrategyBundle(
                    population_strategy(start_camera.dna),
                    fitness_strategy,
                    selection_strategy,
                    crossover_strategy,
                    mutation_strategy,
                    termination_strategy,
                )

                edge_image = self._construct_edge_image(
                    self._image_shape,
                    self._camera_translator,
                    self._target_genome,
                    fitting_geometry,
                    noise_strategy,
                )

                evolution_optimizer = EvolutionOptimizer(
                    strategy_bundle,
                    self._genome_parameters,
                    edge_image,
                    fitting_geometry,
                    headless,
                )

                evolution_result = evolution_optimizer.optimize()
                logger.info(
                    "ResultCode: [{}] {} ({}s)",
                    evolution_result.result_code,
                    strategy_bundle.name_identifier,
                    evolution_result.optimize_duration,
                )

                result_camera = Camera.from_dna(evolution_result.best_result)

                reprojection_errors = reprojection_error_multiple_geometries(
                    start_camera, result_camera.dna, evaluation_geometries
                )

                fitting_geometry_result = reprojection_errors[0]
                dense_geometry_result = reprojection_errors[1]
                y0_geometry_result = reprojection_errors[2]

                n_generations = evolution_result.n_generations
                best_fitness = evolution_result.best_fitness

                self._evolution_writer.save_experiment(
                    strategy_bundle,
                    amount,
                    start_camera,
                    self._target_camera,
                    result_camera,
                    noise_strategy,
                    fitting_geometry_result,
                    dense_geometry_result,
                    y0_geometry_result,
                    best_fitness,
                    n_generations,
                )
