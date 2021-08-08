from dataclasses import dataclass
from timeit import default_timer as timer

import cv2 as cv
import nlopt as nlopt
import numpy as np
from evolution.base import BaseGeometry, FitnessStrategy
from evolution.camera import render_geometry_with_camera
from loguru import logger

from optimizer.optimizer import Optimizer, OptimizerResult, OptimizerResultCode
from utils.transform_utils import split_dna


@dataclass
class NloptAlgorithm:
    display_name: str
    value: int
    needs_bounds: bool = False
    local_optimizer: "NloptAlgorithm" = None


class NloptAlgorithms:
    L_SBPLX = NloptAlgorithm("L_SBPLX", nlopt.LN_SBPLX, False)
    L_NELDER = NloptAlgorithm("L_NELDER", nlopt.LN_NELDERMEAD, False)
    L_COBYLA = NloptAlgorithm("L_COBYLA", nlopt.LN_COBYLA, False)

    G_ISRES = NloptAlgorithm("G_ISRES", nlopt.GN_ISRES, True, L_SBPLX)

    G_CRS = NloptAlgorithm("G_CRS", nlopt.GN_CRS2_LM, True, L_SBPLX)
    G_DIRECT_L = NloptAlgorithm("G_DIRECT_L", nlopt.GN_DIRECT_L_RAND, True, L_SBPLX)
    G_ESCH = NloptAlgorithm("G_ESCH", nlopt.GN_ESCH, True, L_SBPLX)


class NloptOptimizer(Optimizer):
    def __init__(
            self,
            fitness_strategy: FitnessStrategy,
            edge_image: np.array,
            start_dna: np.array,
            geometry: BaseGeometry,
            nlopt_algorithm: NloptAlgorithm,
            headless: bool = True,
    ):
        super().__init__()
        _N_CAMERA_PARAMETERS = 15  # TODO: Derive from len(start_genome)
        _OPTIMIZER_ALGO = nlopt_algorithm.value
        _LOCAL_OPTIMIZER_ALGO = nlopt.LN_NELDERMEAD
        _needs_local_optimizer = nlopt_algorithm.local_optimizer is not None

        _needs_bounds = nlopt_algorithm.needs_bounds
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

            fitness = fitness_lookup.sum().astype(np.float64)

            if fitness > self._best_fitness:
                self._best_fitness = fitness

            return fitness

        self._optimizer.set_max_objective(evo_fitness_fn)

        # logger.warning("MAX TIME IS SET TO 2!")
        # self._optimizer.set_maxtime(20)
        self._optimizer.set_maxeval(10000)

        if _needs_bounds:
            logger.warning("Enabled bounds")
            self._optimizer.set_lower_bounds(
                [500, 500, 200, 100,
                 -3.2, 0.1, 1.0,
                 0.0, -0.2, -0.2,
                 -0.5, -0.5, -0.2, -0.1, -3.0]
            )
            self._optimizer.set_upper_bounds(
                [900, 900, 600, 500,
                 3.2, 8.0, 10.0,
                 0.5, 0.2, 0.2,
                 0.5, 0.5, 0.2, 0.1, 3.0]
            )

        if _needs_local_optimizer:
            _LOCAL_OPTIMIZER_ALGO = nlopt_algorithm.local_optimizer.value
            local_opt = nlopt.opt(_LOCAL_OPTIMIZER_ALGO, _N_CAMERA_PARAMETERS)
            local_opt.set_max_objective(evo_fitness_fn)
            self._optimizer.set_local_optimizer(local_opt)

        self._start_dna = start_dna

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
        result_data = self._optimizer.optimize(self._start_dna.astype(np.float64))
        end_time = timer()

        duration_in_s = end_time - start_time

        result_code = self._optimizer.last_optimize_result()
        result_value = self._optimizer.last_optimum_value()
        mapped_result_code = self.map_result_code(result_code)

        return OptimizerResult(
            result_data, duration_in_s, np.zeros(1), -1, self._best_fitness, mapped_result_code
        )
