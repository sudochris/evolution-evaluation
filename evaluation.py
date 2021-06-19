# region [ImportColor] Import
import numpy as np
from evolution.camera import (
    CameraGenomeParameters,
    ObjGeometry,
)
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

import scripts.mp_evaluator_evolution as mp_evolution
from cameras.cameras import Amount
from optimizer.nlopt_optimizer import NloptAlgorithms
from scripts.evaluator_nlopt import NloptEvaluator
from utils.noise_utils import GridNoise, HLinesNoise, VLinesNoise, NoNoise, SaltNoise

# endregion
if __name__ == "__main__":
    # region [Region0] (A) General application and logging setup
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    logger.add("application.log", rotation="20 MB", format=fmt)
    # endregion

    # region [Region1] (B) General experiment setup

    parameters_file = "data/squash/parameters.json"
    geometry_file = "data/squash/geometries/squash_court.obj"
    evolution_results_file = "results/mp_evolution_experiments.csv"
    nlopt_results_file = "results/nlopt_experiments_dev.csv.dev"

    image_shape = (600, 800)
    genome_parameters = CameraGenomeParameters(parameters_file, image_shape)
    fitting_geometry = ObjGeometry(geometry_file)
    # endregion

    # region [Region2] (C) Define lists for constructing strategy bundles
    amounts = [Amount.near(), Amount.medium(), Amount.far()]

    population_strategies = [
        ValueUniformPopulation(8),
        ValueUniformPopulation(16),
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

    np.set_printoptions(formatter={'float': '{:0.3f}'.format}, linewidth=np.inf)
    print("Min: {}".format(mutation_strategies[0].mutation_min))
    print("Max: {}".format(mutation_strategies[0].mutation_max))
    print("P  : {}".format(mutation_strategies[0].mutation_probability))
    print("=======")

    print("Min: {}".format(mutation_strategies[1].mutation_min))
    print("Max: {}".format(mutation_strategies[1].mutation_max))
    print("P  : {}".format(mutation_strategies[1].mutation_probability))
    print("D  : {}".format(mutation_strategies[1].distributions))

    run_evolution, run_nlopt = True, False
    # region [Region3] Perform Evolution experiments
    if run_evolution:
        # Run multiprocessing evolution
        mp_evolution.evaluate(image_shape, genome_parameters, evolution_results_file, fitting_geometry, amounts,
                              population_strategies, fitness_strategies, selection_strategies, crossover_strategies,
                              mutation_strategies, termination_strategies, noise_strategies, append_mode=True,
                              headless=True)
    # endregion

    # region [Region4] Perform Nlopt experiments
    nlopt_algorithms = [NloptAlgorithms.G_DIRECT_L]

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
            runs_per_bundle=32,
            headless=False,
        )
    # endregion
