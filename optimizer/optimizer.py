from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from loguru import logger
from numpy.core.defchararray import array


class OptimizerResultCode(IntEnum):
    UNKNOWN = 0
    SUCCESS = 1
    STOPVAL_REACHED = 2
    FTOL_REACHED = 3
    XTOL_REACHED = 4
    MAXEVAL_REACHED = 5
    MAXTIME_REACHED = 6
    FAILURE = -1
    INVALID_ARGS = -2
    OUT_OF_MEMORY = -3
    ROUNDOFF_LIMITED = -4
    FORCED_STOP = -5


@dataclass
class OptimizerResult:
    """[summary]"""

    best_result: np.array  # The actual dna
    optimize_duration: float  # Time in seconds
    fitness_history: np.array  # best fitness history
    n_generations: int  # Number of generations if applicable
    best_fitness: float  # Best fitness
    result_code: OptimizerResultCode  # Result code


class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self) -> OptimizerResult:
        raise NotImplementedError("Currently not implemented")

    @abstractmethod
    def map_result_code(self, value) -> OptimizerResultCode:
        raise NotImplementedError("Currently not implemented")
