import itertools
from typing import Callable

import numpy as np
from cameras.cameras import Amount, Camera, wiggle_camera
from evolution.base import (
    BaseGeometry,
    CrossoverStrategy,
    FitnessStrategy,
    MutationStrategy,
    PopulateStrategy,
    SelectionStrategy,
    TerminationStrategy,
)
from evolution.base.base_geometry import DenseGeometry, PlaneGeometry
from evolution.camera import CameraGenomeParameters
from evolution.strategies import StrategyBundle
from loguru import logger
from optimizer.evolution_optimizer import EvolutionOptimizer
from utils.error_utils import reprojection_error_multiple_geometries
from utils.noise_utils import NoiseStrategy
from utils.persistence_utils import EvolutionResultWriter

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

        self._evolution_writer: EvolutionResultWriter = EvolutionResultWriter(
            output_file, append_mode=append_mode
        )

    def evaluate(
        self,
        fitting_geometry: BaseGeometry,
        amounts: list[Amount],
        population_strategies: list[Callable[[np.array], PopulateStrategy]],
        fitness_strategies: list[FitnessStrategy],
        selection_strategies: list[SelectionStrategy],
        crossover_strategies: list[CrossoverStrategy],
        mutation_strategies: list[MutationStrategy],
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
            mutation_strategies,
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
            the_population_strategy = population_strategy(np.zeros(15))
            n_performed_experiments = self._evolution_writer.has(
                amount,
                the_population_strategy,
                fitness_strategy,
                selection_strategy,
                crossover_strategy,
                mutation_strategy,
                noise_strategy,
            )
            if n_performed_experiments > 0:
                printable_experiment_string = "{} {} {} {} {} {} {}".format(
                    amount.name,
                    the_population_strategy.printable_identifier(),
                    fitness_strategy.printable_identifier(),
                    selection_strategy.printable_identifier(),
                    crossover_strategy.printable_identifier(),
                    mutation_strategy.printable_identifier(),
                    noise_strategy.printable_identifier(),
                )
                delta = runs_per_bundle - n_performed_experiments
                logger.info(
                    f"Found {n_performed_experiments} experiments {delta} missing [{printable_experiment_string}] "
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
                    self._target_camera,
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

                evolution_result = evolution_optimizer.optimize(start_camera.dna)
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
