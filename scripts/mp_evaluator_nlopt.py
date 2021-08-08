import itertools
import time
from multiprocessing import Lock, Pool
from typing import Tuple, List

import numpy as np
from evolution.base import BaseGeometry, FitnessStrategy, DenseGeometry, PlaneGeometry
from evolution.camera import CameraGenomeParameters, render_geometry_with_camera
from loguru import logger

from cameras.cameras import Amount, mean_squash_camera, wiggle_camera, Camera
from optimizer.nlopt_optimizer import NloptAlgorithms, NloptOptimizer, NloptAlgorithm
from utils.color_utils import Color
from utils.error_utils import reprojection_error_multiple_geometries
from utils.mp_persistence import MPNloptResultWriter
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


def _do_optimization(amount: Amount,
                     fitness_strategy: FitnessStrategy,
                     nlopt_algorithm: NloptAlgorithm,
                     noise_strategy: NoiseStrategy):
    results_file = shared_data["results_file"]
    fitting_geometry = shared_data["fitting_geometry"]
    headless = shared_data["headless"]
    target_camera = shared_data["target_camera"]
    image_shape = shared_data["image_shape"]
    genome_parameters = shared_data["genome_parameters"]
    nlopt_writer: MPNloptResultWriter = shared_data["nlopt_writer"]

    algorithm_name = nlopt_algorithm.display_name

    _wiggle_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    _n_wiggles = 7
    _runs_per_bundle = 32
    geometry_dense = DenseGeometry(fitting_geometry, 16)
    geometry_y0 = PlaneGeometry(fitting_geometry, 0, 16)
    evaluation_geometries = [fitting_geometry, geometry_dense, geometry_y0]

    printable_experiment_string = "{} {} {} {}".format(amount.name,
                                                       fitness_strategy.printable_identifier(),
                                                       nlopt_algorithm.display_name,
                                                       noise_strategy.printable_identifier())

    with shared_data["file_lock"]:
        n_performed_experiments = nlopt_writer.has(amount, nlopt_algorithm, fitness_strategy, noise_strategy)

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

        nlopt_optimizer = NloptOptimizer(
            fitness_strategy,
            edge_image,
            start_camera.dna,
            fitting_geometry,
            nlopt_algorithm,
            headless,
        )
        try:
            nlopt_result = nlopt_optimizer.optimize()
        except ValueError as e:
            with shared_data["print_lock"]:
                logger.error(e)
            continue
            
        with shared_data["print_lock"]:
            logger.info(
                "{} took {} [{}]", algorithm_name, nlopt_result.optimize_duration, nlopt_result.result_code
            )
        results.append(nlopt_result)
        results_start_camera.append(start_camera)

    results_result_camera = []
    results_fitting_geometry_result = []
    results_dense_geometry_result = []
    results_y0_geometry_result = []
    results_duration_in_s = []
    results_best_fitness = []

    for result in results:
        result_camera = Camera.from_dna(result.best_result)

        reprojection_errors = reprojection_error_multiple_geometries(
            target_camera, result_camera.dna, evaluation_geometries
        )

        fitting_geometry_result = reprojection_errors[0]
        dense_geometry_result = reprojection_errors[1]
        y0_geometry_result = reprojection_errors[2]

        best_fitness = result.best_fitness
        duration_in_s = result.optimize_duration

        results_result_camera.append(result_camera)

        results_fitting_geometry_result.append(fitting_geometry_result)
        results_dense_geometry_result.append(dense_geometry_result)
        results_y0_geometry_result.append(y0_geometry_result)

        results_duration_in_s.append(duration_in_s)
        results_best_fitness.append(best_fitness)

    with shared_data["file_lock"]:
        nlopt_writer.save_experiments(
            algorithm_name,
            fitness_strategy,
            amount,
            results_start_camera,
            target_camera,
            results_result_camera,
            noise_strategy,
            results_fitting_geometry_result,
            results_dense_geometry_result,
            results_y0_geometry_result,
            results_best_fitness,
            results_duration_in_s
        )


def evaluate(image_shape: Tuple[int, int],
             genome_parameters: CameraGenomeParameters,
             nlopt_results_file: str,
             fitting_geometry: BaseGeometry,
             amounts: List[Amount],
             fitness_strategies: List[FitnessStrategy],
             nlopt_algorithms: List[NloptAlgorithms],
             noise_strategies: List[NoiseStrategy],
             append_mode=False,
             headless=True):
    nlopt_writer = MPNloptResultWriter(nlopt_results_file, append_mode=append_mode)

    target_camera = mean_squash_camera(image_shape)

    arguments = [
        amounts,
        fitness_strategies,
        nlopt_algorithms,
        noise_strategies
    ]

    shared = {
        "file_lock": Lock(),
        "print_lock": Lock(),
        "results_file": nlopt_results_file,
        "image_shape": image_shape,
        "target_camera": target_camera,
        "fitting_geometry": fitting_geometry,
        "genome_parameters": genome_parameters,
        "headless": headless,
        "nlopt_writer": nlopt_writer
    }

    start = time.perf_counter()
    n_processes = 10
    with Pool(processes=n_processes, initializer=init, initargs=(shared,)) as pool:
        pool.starmap(_do_optimization, itertools.product(*arguments))
    end = time.perf_counter()

    print(f"All nlopt evaluations in {end - start:0.4f} seconds")
