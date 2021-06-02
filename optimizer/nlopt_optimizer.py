from timeit import default_timer as timer

import cv2 as cv
import nlopt as nlopt
import numpy as np
from evolution.base import BaseGeometry, FitnessStrategy
from evolution.camera import render_geometry_with_camera
from loguru import logger

from optimizer.optimizer import Optimizer, OptimizerResult, OptimizerResultCode
from utils.transform_utils import split_dna


class NloptAlgorithms:
    """All defined algorithms are on purpose derivative free"""

    # region [Region0] Global Algorithms
    # G_DIRECT = nlopt.GN_DIRECT
    # endregion
    # region [Region3] Local Algorithms
    L_COBYLA = nlopt.LN_COBYLA
    # L_BOBYQA = nlopt.LN_BOBYQA
    L_SBPLX = nlopt.LN_SBPLX

    # endregion

    @staticmethod
    def get_algorithm_name(value) -> str:
        return {
            NloptAlgorithms.L_SBPLX: "L_SBPLX",
            NloptAlgorithms.L_COBYLA: "L_COBYLA",
        }.get(value, "UNKNOWN")


class NloptOptimizer(Optimizer):
    def __init__(
            self,
            fitness_strategy: FitnessStrategy,
            edge_image: np.array,
            start_dna: np.array,
            geometry: BaseGeometry,
            nlopt_algorithm,
            headless: bool = True,
    ):
        super().__init__()
        _N_CAMERA_PARAMETERS = 15  # TODO: Derive from len(start_genome)
        _OPTIMIZER_ALGO = nlopt_algorithm
        _LOCAL_OPTIMIZER_ALGO = nlopt.LN_NELDERMEAD
        _needs_local_optimizer = False

        self._optimizer = nlopt.opt(_OPTIMIZER_ALGO, _N_CAMERA_PARAMETERS)

        _fitness_map = fitness_strategy.create_fitness(edge_image)
        _render_image = np.zeros_like(edge_image)
        _geometry = geometry

        self._best_fitness = 0

        def evo_fitness_fn(x, grad):
            _render_image[:] = 0
            camera_matrix, t_vec, r_vec, d_vec = split_dna(x)
            render_geometry_with_camera(
                _render_image, _geometry, camera_matrix, t_vec, r_vec, d_vec, (255,), 2
            )
            fitness_lookup = cv.bitwise_and(_fitness_map, _fitness_map, mask=_render_image)

            if not headless:
                cv.imshow("I1", _render_image)
                cv.imshow("I2", fitness_lookup)
                cv.imshow("I3", _fitness_map)
                cv.waitKey(1)

            fitness = fitness_lookup.sum().astype(float)

            if fitness > self._best_fitness:
                self._best_fitness = fitness

            return fitness

        self._optimizer.set_max_objective(evo_fitness_fn)
        logger.warning("MAX TIME IS SET TO 2!")
        self._optimizer.set_maxtime(2)
        # self._optimizer.set_lower_bounds(
        #     [500, 500, 200, 100, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0]
        # )
        # self._optimizer.set_upper_bounds(
        #     [900, 900, 600, 500, 2.0, 8.0, 15.0, 0.5, 0, 0, 0, 0, 0, 0, 0]
        # )

        if _needs_local_optimizer:
            local_opt = nlopt.opt(_LOCAL_OPTIMIZER_ALGO, _N_CAMERA_PARAMETERS)
            local_opt.set_max_objective(evo_fitness_fn)
            self._optimizer.set_local_optimizer(local_opt)

        self._start_dna = start_dna

        logger.debug(self._optimizer.get_algorithm_name())

    def map_result_code(self, value) -> OptimizerResultCode:
        return {
            nlopt.SUCCESS: OptimizerResultCode.SUCCESS,
            nlopt.STOPVAL_REACHED: OptimizerResultCode.STOPVAL_REACHED,
            nlopt.FTOL_REACHED: OptimizerResultCode.FTOL_REACHED,
            nlopt.XTOL_REACHED: OptimizerResultCode.XTOL_REACHED,
            nlopt.MAXEVAL_REACHED: OptimizerResultCode.MAXEVAL_REACHED,
            nlopt.MAXTIME_REACHED: OptimizerResultCode.MAXTIME_REACHED,
            nlopt.FAILURE: OptimizerResultCode.FAILURE,
            nlopt.INVALID_ARGS: OptimizerResultCode.INVALID_ARGS,
            nlopt.OUT_OF_MEMORY: OptimizerResultCode.OUT_OF_MEMORY,
            nlopt.ROUNDOFF_LIMITED: OptimizerResultCode.ROUNDOFF_LIMITED,
            nlopt.FORCED_STOP: OptimizerResultCode.FORCED_STOP,
        }.get(value, OptimizerResultCode.UNKNOWN)

    def optimize(self) -> OptimizerResult:
        logger.debug("Running nlopt optimizer")

        start_time = timer()
        result_data = self._optimizer.optimize(self._start_dna)
        end_time = timer()

        duration_in_s = end_time - start_time

        result_code = self._optimizer.last_optimize_result()
        result_value = self._optimizer.last_optimum_value()
        mapped_result_code = self.map_result_code(result_code)

        return OptimizerResult(
            result_data, duration_in_s, np.zeros(1), -1, self._best_fitness, mapped_result_code
        )
