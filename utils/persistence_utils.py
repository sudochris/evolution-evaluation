from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import prefixed
from cameras.cameras import Amount, Camera, mean_squash_camera
from evolution.strategies import StrategyBundle, strategy_bundle
from loguru import logger

from utils.error_utils import ReprojectionErrorResult
from utils.exceptions import HeaderValueMismatch, InvalidHeader


class ResultWriter(ABC):
    def __init__(self, outfile, n_flush=1, append_mode=True):
        super().__init__()
        self._outfile = outfile
        self._append_mode = append_mode
        self._n_flush = n_flush
        self._flush_counter = n_flush

        header = self._header()

        if Path(self._outfile).is_file() and append_mode:
            logger.info(f"Open file '{self._outfile}' for appending the results")
            self._data_df = pd.read_csv(self._outfile)
            loaded_header = self._data_df.columns
            if len(loaded_header) != len(header):
                raise InvalidHeader(
                    "Deserialized header length does not match the specified header"
                )
            r = [_lh == _h for _lh, _h in zip(loaded_header, header)]
            all_true = reduce((lambda x, y: x and y), r)
            if not all_true:
                raise InvalidHeader("Deserialized header values do not match the specified header")
        else:
            logger.info(f"Create or overwrite '{self._outfile}' for writing the results")
            self._data_df = pd.DataFrame(columns=header)
            self._data_df.to_csv(self._outfile, index=False)

    def _save_experiment(self, new_row: dict) -> bool:
        self._data_df = self._data_df.append(new_row, ignore_index=True)
        self._flush_counter -= 1
        if self._flush_counter <= 0:
            self.flush()
            return True
        return False

    def flush(self):
        self._data_df.to_csv(self._outfile, index=False)
        self._flush_counter = self._n_flush

    @abstractmethod
    def _header(self) -> tuple[str]:
        pass

    def _get_camera_header_with_prefix(self, prefix: str):
        _intrinsic = ("fu", "fv", "cx", "cy")
        _extrinsic = ("tx", "ty", "tz", "rx", "ry", "rz")
        _distortion = ("d0", "d1", "d2", "d3", "d4")
        _full_parameters = _intrinsic + _extrinsic + _distortion
        return tuple([f"{prefix}_{v}" for v in _full_parameters])

    def _construct_camera_entries(self, prefix: str, camera: Camera) -> dict:
        _intrinsic = ("fu", "fv", "cx", "cy")
        _extrinsic = ("tx", "ty", "tz", "rx", "ry", "rz")
        _distortion = ("d0", "d1", "d2", "d3", "d4")
        _full_parameters = _intrinsic + _extrinsic + _distortion

        return {f"{prefix}_{k}": v for k, v in zip(_full_parameters, camera.dna)}

    def _get_geometry_header_width_infix(self, infix: str):
        return (f"mean_{infix}_reproj_error_col", f"mean_{infix}_reproj_error_row")

    def _construct_geometry_entries(self, infix: str, result: ReprojectionErrorResult) -> dict:
        return {
            f"mean_{infix}_reproj_error_col": result.mean_col_error,
            f"mean_{infix}_reproj_error_row": result.mean_row_error,
        }


class EvolutionResultWriter(ResultWriter):
    def _header(self) -> tuple[str]:
        # region [Region0] Strategy + Noise Header
        strategy_columns = (
            "population_fn",
            "fitness_fn",
            "selection_fn",
            "crossover_fn",
            "mutation_fn",
            "termination_fn",
            "distance_type",
        )
        noise_columns = ("noise_type", "noise_value")
        # endregion

        # region [Region1] Start, Target and Result Camera Header
        start_camera_header = self._get_camera_header_with_prefix("s")
        target_camera_header = self._get_camera_header_with_prefix("t")
        result_camera_header = self._get_camera_header_with_prefix("r")
        # endregion

        # region [Region2] Fitness + Generation Header
        results = ("best_fitness", "generations")
        # endregion

        # region [Region3] Result for various geometries
        fitting_geometry_result = self._get_geometry_header_width_infix("fitting")
        dense_geometry_result = self._get_geometry_header_width_infix("dense")
        y0_geometry_result = self._get_geometry_header_width_infix("y0")
        results_geometries = fitting_geometry_result + dense_geometry_result + y0_geometry_result
        # endregion

        # region [Region4] Concatenate all results to form the final header
        full_columns = (
            strategy_columns
            + noise_columns
            + start_camera_header
            + target_camera_header
            + result_camera_header
            + results
            + results_geometries
        )
        # endregion
        return full_columns

    def save_experiment(
        self,
        strategy_bundle: StrategyBundle,
        distance_amount: Amount,
        start_camera: Camera,
        target_camera: Camera,
        result_camera: Camera,
        noise_type: str,
        noise_value: float,
        fitting_result: ReprojectionErrorResult,
        dense_result: ReprojectionErrorResult,
        y0_result: ReprojectionErrorResult,
        best_fitness: float,
        generations: int,
    ):
        new_row = {
            "population_fn": strategy_bundle.populate_strategy.printable_identifier(),
            "fitness_fn": strategy_bundle.fitness_strategy.printable_identifier(),
            "selection_fn": strategy_bundle.selection_strategy.printable_identifier(),
            "crossover_fn": strategy_bundle.crossover_strategy.printable_identifier(),
            "mutation_fn": strategy_bundle.mutation_strategy.printable_identifier(),
            "termination_fn": strategy_bundle.termination_strategy.printable_identifier(),
            "distance_type": distance_amount.name,
            "noise_type": noise_type,
            "noise_value": noise_value,
            "best_fitness": best_fitness,
            "generations": generations,
        }

        new_row.update(self._construct_camera_entries("s", start_camera))
        new_row.update(self._construct_camera_entries("t", target_camera))
        new_row.update(self._construct_camera_entries("r", result_camera))

        new_row.update(self._construct_geometry_entries("fitting", fitting_result))
        new_row.update(self._construct_geometry_entries("dense", dense_result))
        new_row.update(self._construct_geometry_entries("y0", y0_result))

        self._save_experiment(new_row)


class NloptResultWriter(ResultWriter):
    def _header(self) -> tuple[str]:

        # region [Region0] Nlopt Specification + Noise Header
        nlopt_type = ("nlopt_optimizer", "distance_type")
        noise_columns = ("noise_type", "noise_value")
        # endregion

        # region [Region1] Start, Target and Result Camera Header
        start_camera_header = self._get_camera_header_with_prefix("s")
        target_camera_header = self._get_camera_header_with_prefix("t")
        result_camera_header = self._get_camera_header_with_prefix("r")
        # endregion

        # region [Region2] Fitness + Generation Header
        results = ("best_fitness",)
        # endregion

        # region [Region3] Result for various geometries
        fitting_geometry_result = self._get_geometry_header_width_infix("fitting")
        dense_geometry_result = self._get_geometry_header_width_infix("dense")
        y0_geometry_result = self._get_geometry_header_width_infix("y0")
        results_geometries = fitting_geometry_result + dense_geometry_result + y0_geometry_result
        # endregion

        # region [Region4] Concatenate all results to form the final header
        full_columns = (
            nlopt_type
            + noise_columns
            + start_camera_header
            + target_camera_header
            + result_camera_header
            + results
            + results_geometries
        )
        # endregion
        return full_columns

    def save_experiment(
        self,
        nlopt_optimizer: str,
        distance_amount: Amount,
        start_camera: Camera,
        target_camera: Camera,
        result_camera: Camera,
        noise_type: str,
        noise_value: float,
        fitting_result: ReprojectionErrorResult,
        dense_result: ReprojectionErrorResult,
        y0_result: ReprojectionErrorResult,
        best_fitness: float,
    ):
        new_row = {
            "nlopt_optimizer": nlopt_optimizer,
            "distance_type": distance_amount.name,
            "noise_type": noise_type,
            "noise_value": noise_value,
            "best_fitness": best_fitness,
        }

        new_row.update(self._construct_camera_entries("s", start_camera))
        new_row.update(self._construct_camera_entries("t", target_camera))
        new_row.update(self._construct_camera_entries("r", result_camera))

        new_row.update(self._construct_geometry_entries("fitting", fitting_result))
        new_row.update(self._construct_geometry_entries("dense", dense_result))
        new_row.update(self._construct_geometry_entries("y0", y0_result))

        self._save_experiment(new_row)