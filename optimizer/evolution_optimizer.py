import sys
from timeit import default_timer as timer

import cv2 as cv
import numpy as np
from evolution.base import BaseGeometry
from evolution.camera import CameraGenomeParameters, GeneticCameraAlgorithm
from evolution.strategies import StrategyBundle
from loguru import logger

from optimizer.optimizer import Optimizer, OptimizerResult, OptimizerResultCode


class EvolutionOptimizer(Optimizer):
    def __init__(
            self,
            strategy_bundle: StrategyBundle,
            genome_parameters: CameraGenomeParameters,
            edge_image: np.array,
            geometry: BaseGeometry,
            headless: bool = True,
    ):
        super().__init__()
        self._strategy_bundle = strategy_bundle
        self._evolution = GeneticCameraAlgorithm(
            genome_parameters, strategy_bundle, edge_image, geometry, headless
        )
        cv.imshow("TMP", self._evolution._fitness_map)

    def map_result_code(self, value) -> OptimizerResultCode:
        return {
            0: OptimizerResultCode.SUCCESS,
            1: OptimizerResultCode.STOPVAL_REACHED,
            2: OptimizerResultCode.FTOL_REACHED,
            3: OptimizerResultCode.XTOL_REACHED,
            4: OptimizerResultCode.MAXEVAL_REACHED,
            5: OptimizerResultCode.MAXTIME_REACHED,
            -1: OptimizerResultCode.FAILURE,
            -2: OptimizerResultCode.INVALID_ARGS,
            -3: OptimizerResultCode.OUT_OF_MEMORY,
            -4: OptimizerResultCode.ROUNDOFF_LIMITED,
            -5: OptimizerResultCode.FORCED_STOP,
        }.get(value, OptimizerResultCode.UNKNOWN)

    def optimize(self):
        # logger.debug("Running evolution optimizer")
        try:
            start_time = timer()
            result_data = self._evolution.run()
            end_time = timer()
        except ValueError as e:
            logger.error(self._strategy_bundle.name_identifier)
            sys.exit(f"Error evaluating {self._strategy_bundle.name_identifier}")

        duration_in_s = end_time - start_time

        logger.warning("Evolution result code is always SUCCESS!")
        mapped_result_code = self.map_result_code(0)

        best_genome, best_fitness = result_data.best_genome
        best_fitnesses = result_data.best_fitnesses
        n_generations = result_data.n_generations

        return OptimizerResult(
            best_genome.dna,
            duration_in_s,
            best_fitnesses,
            n_generations,
            best_fitness,
            mapped_result_code,
        )
