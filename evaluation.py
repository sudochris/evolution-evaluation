# region [ImportColor] Import
import argparse
import sys

import numpy as np
from evolution.camera import CameraGenomeParameters, ObjGeometry
from evolution.strategies import (
    BoundedDistributionBasedMutation,
    BoundedUniformMutation,
    DistanceMap,
    DistanceMapWithPunishment,
    NoImprovement,
    RouletteWheel,
    SinglePoint,
    Tournament,
    TwoPoint,
    Uniform,
    ValueUniformPopulation,
)
from loguru import logger

from cameras.cameras import Amount
from optimizer.nlopt_optimizer import NloptAlgorithms
from scripts.evaluator_evolution import EvolutionEvaluator
from scripts.evaluator_nlopt import NloptEvaluator
from utils.noise_utils import GridNoise, HLinesNoise, NoNoise, SaltNoise, VLinesNoise

# endregion
if __name__ == "__main__":
    # region [Region0] (A) General application and logging setup
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    logger.add("application.log", rotation="20 MB", format=fmt, level="INFO")
    # endregion

    # region [Region1] (B) General experiment setup
    parser = argparse.ArgumentParser(description="Evolution Resectioning evaluation")
    parser.add_argument(
        "-pf", "--parameters_file", type=str, help="Path to parameters for the algorithm"
    )
    parser.add_argument(
        "-gf", "--geometry_file", type=str, help="Path to parameters for the algorithm"
    )
    parser.add_argument(
        "-erf", "--evolution_results_file", type=str, help="Output path for evolution results"
    )
    parser.add_argument(
        "-nrf", "--nlopt_results_file", type=str, help="Output path for nlopt results"
    )
    parser.add_argument(
        "-evolution", action=argparse.BooleanOptionalAction, help="Run evolution experiments"
    )
    parser.add_argument(
        "-nlopt", action=argparse.BooleanOptionalAction, help="Run NLOPT experiments"
    )
    parser.add_argument(
        "-headless", action=argparse.BooleanOptionalAction, help="Run nlopt experiments"
    )
    parser.add_argument("-rbb", "--runs_per_bundle", type=int, help="Run nlopt experiments")

    args = parser.parse_args()

    run_evolution, run_nlopt, run_headless = False, False, False
    runs_per_bundle = 32

    parameters_file = "data/squash/parameters.json"
    geometry_file = "data/squash/geometries/squash_court.obj"
    evolution_results_file = "evolution_results.csv"  # "results/evolution_experiments_dev.csv.dev"
    nlopt_results_file = "nlopt_results.csv"  # "results/nlopt_experiments_dev.csv.dev"

    if args.parameters_file:
        parameters_file = args.parameters_file
    if args.geometry_file:
        geometry_file = args.geometry_file
    if args.evolution_results_file:
        evolution_results_file = args.evolution_results_file
    if args.nlopt_results_file:
        nlopt_results_file = args.nlopt_results_file
    if args.runs_per_bundle:
        runs_per_bundle = args.runs_per_bundle

    if args.evolution:
        run_evolution = True
    if args.nlopt:
        run_nlopt = True
    if args.headless:
        run_headless = True

    image_shape = (600, 800)
    genome_parameters = CameraGenomeParameters(parameters_file, image_shape)
    fitting_geometry = ObjGeometry(geometry_file)
    # endregion

    # region [Region2] (C) Define lists for constructing strategy bundles
    amounts = [Amount.near(), Amount.medium(), Amount.far()]

    population_strategies = [
        #        lambda start_dna: ValueUniformPopulation(start_dna, 8),
        #        lambda start_dna: ValueUniformPopulation(start_dna, 16),
        lambda start_dna: ValueUniformPopulation(8),
        lambda start_dna: ValueUniformPopulation(16),
    ]

    fitness_strategies = [
        DistanceMapWithPunishment(DistanceMap.DistanceType.L2, 0.3),
        DistanceMap(DistanceMap.DistanceType.L2, 0.3),
    ]

    selection_strategies = [Tournament(4), Tournament(8), RouletteWheel()]

    crossover_strategies = [
        SinglePoint(),
        TwoPoint(),
        Uniform(np.ones(15) * 0.01, "_0.01"),
        Uniform(np.ones(15) * 0.5, "_0.5"),
    ]

    mutation_strategies = [
        BoundedUniformMutation(genome_parameters),
        BoundedDistributionBasedMutation(genome_parameters),
    ]

    termination_strategies = [NoImprovement(300)]

    noise_strategies = [
        NoNoise(),
        SaltNoise(0.01),
        HLinesNoise(spacing=128),
        VLinesNoise(spacing=128),
        GridNoise(hspacing=128, vspacing=128, angle=0),
        GridNoise(hspacing=128, vspacing=128, angle=45),
    ]
    # endregion

    np.set_printoptions(formatter={"float": "{:0.3f}".format}, linewidth=np.inf)
    print("Min: {}".format(mutation_strategies[0].mutation_min))
    print("Max: {}".format(mutation_strategies[0].mutation_max))
    print("P  : {}".format(mutation_strategies[0].mutation_probability))
    print("=======")

    print("Min: {}".format(mutation_strategies[1].mutation_min))
    print("Max: {}".format(mutation_strategies[1].mutation_max))
    print("P  : {}".format(mutation_strategies[1].mutation_probability))
    print("D  : {}".format(mutation_strategies[1].distributions))

    # region [Region3] Perform Evolution experiments
    if run_evolution:
        evaluator = EvolutionEvaluator(
            image_shape, genome_parameters, evolution_results_file, append_mode=True
        )
        evaluator.evaluate(
            fitting_geometry,
            amounts,
            population_strategies,
            fitness_strategies,
            selection_strategies,
            crossover_strategies,
            mutation_strategies,
            termination_strategies,
            noise_strategies,
            runs_per_bundle=runs_per_bundle,
            headless=run_headless,
        )
    # endregion

    # region [Region4] Perform Nlopt experiments
    nlopt_algorithms = [NloptAlgorithms.L_SBPLX, NloptAlgorithms.G_DIRECT_L]

    if run_nlopt:
        nlopt_evaluator = NloptEvaluator(
            image_shape, genome_parameters, nlopt_results_file, append_mode=True
        )
        nlopt_evaluator.evaluate(
            fitting_geometry,
            amounts,
            fitness_strategies,
            nlopt_algorithms,
            noise_strategies,
            runs_per_bundle=runs_per_bundle,
            headless=run_headless,
        )
    # endregion
