import itertools
import time
from multiprocessing import Pool, Lock
from typing import List, Tuple

import numpy as np
from evolution.base import BaseGeometry, PopulateStrategy, FitnessStrategy, SelectionStrategy, CrossoverStrategy, \
    MutationStrategy, TerminationStrategy, DenseGeometry, PlaneGeometry
from evolution.camera import CameraGenomeParameters, render_geometry_with_camera
from evolution.strategies import StrategyBundle
from loguru import logger

from cameras.cameras import Amount, wiggle_camera, Camera, mean_squash_camera
from optimizer.evolution_optimizer import EvolutionOptimizer
from utils.color_utils import Color
from utils.error_utils import reprojection_error_multiple_geometries
from utils.mp_persistence import MPEvolutionResultWriter
from utils.noise_utils import NoiseStrategy, add_noise_to_image
from utils.transform_utils import split_dna


def init(shared):
    global shared_data
    shared_data = shared


def _construct_edge_image(
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


def _do_optimization(amount: Amount, population_strategy: PopulateStrategy,
                     fitness_strategy: FitnessStrategy,
                     selection_strategy: SelectionStrategy, crossover_strategy: CrossoverStrategy,
                     mutation_strategy: MutationStrategy, termination_strategy: TerminationStrategy,
                     noise_strategy: NoiseStrategy):
    results_file = shared_data["results_file"]
    fitting_geometry = shared_data["fitting_geometry"]
    headless = shared_data["headless"]
    target_camera = shared_data["target_camera"]
    image_shape = shared_data["image_shape"]
    genome_parameters = shared_data["genome_parameters"]
    evolution_writer: MPEvolutionResultWriter = shared_data["evolution_writer"]

    # region CONFIGURATION
    _wiggle_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    _n_wiggles = 7
    _runs_per_bundle = 32
    geometry_dense = DenseGeometry(fitting_geometry, 16)
    geometry_y0 = PlaneGeometry(fitting_geometry, 0, 16)
    evaluation_geometries = [fitting_geometry, geometry_dense, geometry_y0]
    # endregion

    strategy_bundle = StrategyBundle(
        population_strategy,
        fitness_strategy,
        selection_strategy,
        crossover_strategy,
        mutation_strategy,
        termination_strategy,
    )
    printable_experiment_string = "{} {} {} {} {} {} {}".format(amount.name,
                                                                population_strategy.printable_identifier(),
                                                                fitness_strategy.printable_identifier(),
                                                                selection_strategy.printable_identifier(),
                                                                crossover_strategy.printable_identifier(),
                                                                mutation_strategy.printable_identifier(),
                                                                noise_strategy.printable_identifier())
    with shared_data["file_lock"]:
        n_performed_experiments = evolution_writer.has(amount, strategy_bundle, noise_strategy)

    if n_performed_experiments > 0:
        delta = _runs_per_bundle - n_performed_experiments
        with shared_data["print_lock"]:
            logger.info(
                f"Found {n_performed_experiments} experiments {delta} missing [{printable_experiment_string}] "
            )
    results = []
    results_start_camera = []
    for _ in range(n_performed_experiments, _runs_per_bundle):
        start_camera = wiggle_camera(
            target_camera, amount, _wiggle_indices, _n_wiggles
        )

        edge_image = _construct_edge_image(
            image_shape,
            target_camera,
            fitting_geometry,
            noise_strategy,
        )

        evolution_optimizer = EvolutionOptimizer(
            strategy_bundle,
            genome_parameters,
            edge_image,
            fitting_geometry,
            headless
        )
        try:
            evolution_result = evolution_optimizer.optimize(start_camera)
        except ValueError as e:
            with shared_data["print_lock"]:
                logger.error(e)
            continue

        with shared_data["print_lock"]:
            logger.info(
                "ResultCode: [{}] {} ({}s)",
                evolution_result.result_code,
                strategy_bundle.name_identifier,
                evolution_result.optimize_duration,
            )
        results.append(evolution_result)
        results_start_camera.append(start_camera)

    results_result_camera = []
    results_fitting_geometry_result = []
    results_dense_geometry_result = []
    results_y0_geometry_result = []
    results_n_generations = []
    results_best_fitness = []

    for result in results:
        result_camera = Camera.from_dna(result.best_result)

        reprojection_errors = reprojection_error_multiple_geometries(
            target_camera, result_camera.dna, evaluation_geometries
        )

        fitting_geometry_result = reprojection_errors[0]
        dense_geometry_result = reprojection_errors[1]
        y0_geometry_result = reprojection_errors[2]

        n_generations = result.n_generations
        best_fitness = result.best_fitness

        results_result_camera.append(result_camera)

        results_fitting_geometry_result.append(fitting_geometry_result)
        results_dense_geometry_result.append(dense_geometry_result)
        results_y0_geometry_result.append(y0_geometry_result)

        results_n_generations.append(n_generations)
        results_best_fitness.append(best_fitness)

    with shared_data["file_lock"]:
        evolution_writer.save_experiments(
            strategy_bundle,
            amount,
            results_start_camera,
            target_camera,
            results_result_camera,
            noise_strategy,
            results_fitting_geometry_result,
            results_dense_geometry_result,
            results_y0_geometry_result,
            results_best_fitness,
            results_n_generations,
        )
    with shared_data["print_lock"]:
        logger.info(f"Success {printable_experiment_string}")


def evaluate(image_shape: Tuple[int, int],
             genome_parameters: CameraGenomeParameters,
             evolution_results_file: str,
             fitting_geometry: BaseGeometry,
             amounts: List[Amount],
             population_strategies: List[PopulateStrategy],
             fitness_strategies: List[FitnessStrategy],
             selection_strategies: List[SelectionStrategy],
             crossover_strategies: List[CrossoverStrategy],
             mutation_strategies: List[MutationStrategy],
             termination_strategies: List[TerminationStrategy],
             noise_strategies: List[NoiseStrategy],
             append_mode=False,
             headless=True):
    evolution_writer = MPEvolutionResultWriter(evolution_results_file, append_mode=append_mode)

    target_camera = mean_squash_camera(image_shape)
    logger.warning("MISSING RUNS PER BUNDLE!")
    arguments = [
        amounts,
        population_strategies,
        fitness_strategies,
        selection_strategies,
        crossover_strategies,
        mutation_strategies,
        termination_strategies,
        noise_strategies,
    ]

    shared = {
        "file_lock": Lock(),
        "print_lock": Lock(),
        "results_file": evolution_results_file,
        "image_shape": image_shape,
        "target_camera": target_camera,
        "fitting_geometry": fitting_geometry,
        "genome_parameters": genome_parameters,
        "headless": headless,
        "evolution_writer": evolution_writer
    }
    start = time.perf_counter()
    n_processes = 10
    with Pool(processes=n_processes, initializer=init, initargs=(shared,)) as pool:
        pool.starmap(_do_optimization, itertools.product(*arguments))  # itertools.product(*strategies))
    end = time.perf_counter()

    print(f"All experiments in {end - start:0.4f} seconds")
