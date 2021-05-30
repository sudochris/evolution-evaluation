# region [ImportColor] Import
from typing import Tuple

import cv2 as cv
import nlopt
import numpy as np
from evolution.base import DenseGeometry, PlaneGeometry, SelectionStrategy
from evolution.camera import (
    CameraGenomeFactory,
    CameraGenomeParameters,
    CameraTranslator,
    ObjGeometry,
    render_geometry_with_camera,
)
from evolution.strategies import (
    BoundedUniformMutation,
    DistanceMap,
    DistanceMapWithPunishment,
    NoImprovement,
    StrategyBundle,
    Tournament,
    TwoPoint,
    ValueUniformPopulation,
)
from loguru import logger

from cameras.cameras import Amount, Camera, mean_squash_camera, wiggle_camera
from optimizer.evolution_optimizer import EvolutionOptimizer
from optimizer.nlopt_optimizer import NloptAlgorithms, NloptOptimizer
from utils.color_utils import Color
from utils.error_utils import (
    ReprojectionErrorResult,
    reprojection_error,
    reprojection_error_multiple_geometries,
)
from utils.noise_utils import add_noise_to_image, noise_hlines, noise_salt
from utils.persistence_utils import EvolutionResultWriter, NloptResultWriter, ResultWriter
from utils.transform_utils import split_dna

# endregion


# region [TMP]

if __name__ == "__main222__":

    start_camera = target_camera = result_camera = mean_squash_camera((800, 600))
    noise_type = "h_lines"
    noise_value = 128

    fitting_result = dense_result = y0_result = ReprojectionErrorResult(
        np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), 0, 0, 0
    )

    population_strategy = ValueUniformPopulation(start_camera.dna, 8)
    fitness_strategy = DistanceMapWithPunishment(DistanceMap.DistanceType.L2, 0.3)
    selection_strategy = Tournament(4)
    crossover_strategy = TwoPoint()

    parameters_file = "data/squash/parameters.json"
    image_shape = (600, 800)
    genome_parameters = CameraGenomeParameters(parameters_file, image_shape)
    mutation_strategy = BoundedUniformMutation(genome_parameters)
    termination_strategy = NoImprovement(300)

    strategy_bundle = StrategyBundle(
        population_strategy,
        fitness_strategy,
        selection_strategy,
        crossover_strategy,
        mutation_strategy,
        termination_strategy,
    )

    evolution_writer.save_experiment(
        strategy_bundle,
        Amount.near(),
        start_camera,
        target_camera,
        result_camera,
        noise_type,
        noise_value,
        fitting_result,
        dense_result,
        y0_result,
        1337.4711,
        10023,
    )
    nlopt_writer.save_experiment(
        "LN_SUBPLX",
        Amount.far(),
        start_camera,
        target_camera,
        result_camera,
        noise_type,
        noise_value,
        fitting_result,
        dense_result,
        y0_result,
    )


# endregion


if __name__ == "__main__":
    # region [Region0] (A) General application and logging setup
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    logger.add("application.log", rotation="20 MB", format=fmt)
    # endregion

    # region [Region1] (B) Experiment Parameters
    parameters_file = "data/squash/parameters.json"
    geometry_file = "data/squash/geometries/squash_court.obj"
    image_shape = (600, 800)
    image_height, image_width = image_shape
    geometry = ObjGeometry(geometry_file)

    camera_translator = CameraTranslator()
    genome_parameters = CameraGenomeParameters(parameters_file, image_shape)
    camera_genome_factory = CameraGenomeFactory(genome_parameters)

    draw_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    # endregion

    # region [Region2] (C) 1. Create synthetic camera data
    # The synthetic camera represents the mean extracted camera view
    # from various broadcasted squash videos
    target_camera = mean_squash_camera(image_shape)
    target_genome = camera_genome_factory.create(target_camera.dna, "TargetGenome")
    target_cm, target_t, target_r, target_d = camera_translator.translate_genome(target_genome)

    # Rendering the geometry using the target camera results in an
    # edge image which will be used as input for the fitness lookup map
    #
    edge_image = np.zeros(image_shape, dtype=np.uint8)
    render_geometry_with_camera(
        edge_image, geometry, target_cm, target_t, target_r, target_d, Color.WHITE
    )

    render_geometry_with_camera(
        draw_image, geometry, target_cm, target_t, target_r, target_d, Color.WHITE
    )
    # endregion

    # region [TMP]
    noise_type = "h_lines"
    noise_value = 128
    noise_image = noise_hlines(edge_image, spacing=noise_value)
    edge_image = add_noise_to_image(edge_image, noise_image)
    # endregion

    # region [Region3] (D) Perform the actual experiments
    start_camera = mean_squash_camera(image_shape)
    wiggle_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    n_wiggles = 7
    wiggle_amount = Amount.near()
    start_camera = wiggle_camera(start_camera, wiggle_amount, wiggle_indices, n_wiggles)
    start_genome = camera_genome_factory.create(start_camera.dna, "StartGenome")
    start_cm, start_t, start_r, start_d = camera_translator.translate_genome(start_genome)

    render_geometry_with_camera(
        draw_image, geometry, start_cm, start_t, start_r, start_d, Color.RED, 1
    )
    # endregion

    # region [Region4] (E) Select and construct strategy bundle
    population_strategy = ValueUniformPopulation(start_camera.dna, 8)
    fitness_strategy = DistanceMapWithPunishment(DistanceMap.DistanceType.L2, 0.3)
    selection_strategy = Tournament(4)
    crossover_strategy = TwoPoint()
    mutation_strategy = BoundedUniformMutation(genome_parameters)
    termination_strategy = NoImprovement(32)

    strategy_bundle = StrategyBundle(
        population_strategy,
        fitness_strategy,
        selection_strategy,
        crossover_strategy,
        mutation_strategy,
        termination_strategy,
    )
    # endregion

    nlopt_algorithm = NloptAlgorithms.L_SBPLX
    run_nlopt, run_evo = True, True

    headless = True

    # region [TMP]
    geometry_dense = DenseGeometry(geometry, 16)
    geometry_y0 = PlaneGeometry(geometry, 0, 16)
    # endregion

    evolution_writer: ResultWriter = EvolutionResultWriter("results/evo.csv", append_mode=False)
    nlopt_writer: ResultWriter = NloptResultWriter("results/nlopt.csv", append_mode=False)
    logger.warning("Result writer is OVERWRITING previous results!")

    # 4. Perform optimization experiments
    if run_nlopt:
        nlopt_optimizer = NloptOptimizer(
            fitness_strategy, edge_image, start_camera.dna, geometry, nlopt_algorithm, headless
        )
        nlopt_result = nlopt_optimizer.optimize()
        logger.info("NLOPT {} took {}", nlopt_result.result_code, nlopt_result.optimize_duration)

        result_dna = nlopt_result.best_result
        result_genome = camera_genome_factory.create(result_dna, "ResultGenome")
        result_cm, result_t, result_r, result_d = camera_translator.translate_genome(result_genome)

        render_geometry_with_camera(
            draw_image, geometry, result_cm, result_t, result_r, result_d, Color.BLUE, 3
        )
        nlopt_algorithm_name = NloptAlgorithms.get_algorithm_name(nlopt_algorithm)
        result_camera = Camera.from_dna(nlopt_result.best_result)
        reprojection_errors = reprojection_error_multiple_geometries(
            start_camera, result_dna, [geometry, geometry_dense, geometry_y0]
        )
        fitting_geometry_result, dense_geometry_result, y0_geometry_result = reprojection_errors

        best_fitness = nlopt_result.best_fitness

        nlopt_writer.save_experiment(
            nlopt_algorithm_name,
            wiggle_amount,
            start_camera,
            target_camera,
            result_camera,
            noise_type,
            noise_value,
            fitting_geometry_result,
            dense_geometry_result,
            y0_geometry_result,
            best_fitness,
        )
    if run_evo:
        evolution_optimizer = EvolutionOptimizer(
            strategy_bundle, genome_parameters, edge_image, geometry, headless
        )
        evolution_result = evolution_optimizer.optimize()
        logger.info(
            "EVO {} took {}", evolution_result.result_code, evolution_result.optimize_duration
        )

        result_genome = camera_genome_factory.create(evolution_result.best_result, "ResultGenome")
        result_cm, result_t, result_r, result_d = camera_translator.translate_genome(result_genome)

        render_geometry_with_camera(
            draw_image, geometry, result_cm, result_t, result_r, result_d, Color.GREEN, 2, False
        )

        result_camera = Camera.from_dna(evolution_result.best_result)
        reprojection_errors = reprojection_error_multiple_geometries(
            start_camera, result_dna, [geometry, geometry_dense, geometry_y0]
        )
        fitting_geometry_result, dense_geometry_result, y0_geometry_result = reprojection_errors
        n_generations = evolution_result.n_generations
        best_fitness = evolution_result.best_fitness

        evolution_writer.save_experiment(
            strategy_bundle,
            wiggle_amount,
            start_camera,
            target_camera,
            result_camera,
            noise_type,
            noise_value,
            fitting_geometry_result,
            dense_geometry_result,
            y0_geometry_result,
            best_fitness,
            n_generations,
        )

    # 5. Report result
    #    fitness_map = fitness_strategy.create_fitness(edge_image)
    #    cv.imshow("TEST", fitness_map)
    cv.imshow("TEST", draw_image)
    cv.waitKey(0)
