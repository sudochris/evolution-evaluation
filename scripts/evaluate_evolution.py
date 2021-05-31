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


def _construct_edge_image(
    image_shape: tuple[int, int],
    translator: CameraTranslator,
    genome: BaseGenome,
    fitting_geometry: BaseGeometry,
    noise_strategy: NoiseStrategy,
) -> np.array:
    edge_image = np.zeros(image_shape, dtype=np.uint8)

    cm, t, r, d = translator.translate_genome(genome)
    render_geometry_with_camera(
        edge_image,
        fitting_geometry,
        cm,
        t,
        r,
        d,
        Color.WHITE,
    )

    noise_image = noise_strategy.generate_noise(edge_image)
    return add_noise_to_image(edge_image, noise_image)


def experiment_generator(
    amounts: list[Amount],
    population_strategies: list[PopulateStrategy],
    fitness_strategies: list[FitnessStrategy],
    selection_strategies: list[SelectionStrategy],
    crossover_strategies: list[CrossoverStrategy],
    mutation_stategies: list[MutationStrategy],
    termination_strategies: list[TerminationStrategy],
    noise_strategies: list[NoiseStrategy],
):
    for amount in amounts:
        for population_strategy in population_strategies:
            for fitness_strategy in fitness_strategies:
                for selection_strategy in selection_strategies:
                    for crossover_strategy in crossover_strategies:
                        for mutation_strategy in mutation_stategies:
                            for termination_strategy in termination_strategies:
                                for noise_strategy in noise_strategies:
                                    yield (
                                        amount,
                                        population_strategy,
                                        fitness_strategy,
                                        selection_strategy,
                                        crossover_strategy,
                                        mutation_strategy,
                                        termination_strategy,
                                        noise_strategy,
                                    )


def evaluate_evolutions(
    image_shape: tuple[int, int],
    genome_parameters: CameraGenomeParameters,
    fitting_geometry: BaseGeometry,
    amounts: list[Amount],
    population_strategies: list[PopulateStrategy],
    fitness_strategies: list[FitnessStrategy],
    selection_strategies: list[SelectionStrategy],
    crossover_strategies: list[CrossoverStrategy],
    mutation_stategies: list[MutationStrategy],
    termination_strategies: list[TerminationStrategy],
    noise_strategies: list[NoiseStrategy],
    output_file: str,
    append_mode: True,
    runs_per_bundle=32,
    headless=True,
):

    camera_genome_factory = CameraGenomeFactory(genome_parameters)
    camera_translator = CameraTranslator()

    target_camera = mean_squash_camera(image_shape)
    target_genome = camera_genome_factory.create(target_camera.dna, "TargetGenome")

    wiggle_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    n_wiggles = 7

    evolution_writer: ResultWriter = EvolutionResultWriter(output_file, append_mode=append_mode)

    geometry_dense = DenseGeometry(fitting_geometry, 16)
    geometry_y0 = PlaneGeometry(fitting_geometry, 0, 16)
    evaluation_geometries = [fitting_geometry, geometry_dense, geometry_y0]

    for (
        amount,
        population_strategy,
        fitness_strategy,
        selection_strategy,
        crossover_strategy,
        mutation_strategy,
        termination_strategy,
        noise_strategy,
    ) in experiment_generator(
        amounts,
        population_strategies,
        fitness_strategies,
        selection_strategies,
        crossover_strategies,
        mutation_stategies,
        termination_strategies,
        noise_strategies,
    ):
        for _ in range(runs_per_bundle):
            start_camera = wiggle_camera(target_camera, amount, wiggle_indices, n_wiggles)

            strategy_bundle = StrategyBundle(
                population_strategy(start_camera.dna),
                fitness_strategy,
                selection_strategy,
                crossover_strategy,
                mutation_strategy,
                termination_strategy,
            )

            edge_image = _construct_edge_image(
                image_shape, camera_translator, target_genome, fitting_geometry, noise_strategy
            )

            evolution_optimizer = EvolutionOptimizer(
                strategy_bundle, genome_parameters, edge_image, fitting_geometry, headless
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

            evolution_writer.save_experiment(
                strategy_bundle,
                amount,
                start_camera,
                target_camera,
                result_camera,
                noise_strategy,
                fitting_geometry_result,
                dense_geometry_result,
                y0_geometry_result,
                best_fitness,
                n_generations,
            )
