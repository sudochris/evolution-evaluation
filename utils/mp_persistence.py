from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from evolution.base import FitnessStrategy
from evolution.strategies import StrategyBundle
from loguru import logger

from cameras.cameras import Amount, Camera
from optimizer.nlopt_optimizer import NloptAlgorithm
from utils.error_utils import ReprojectionErrorResult
from utils.exceptions import InvalidHeader
from utils.noise_utils import NoiseStrategy


class MPResultWriter(ABC):
    def __init__(self, outfile, append_mode=True):
        super().__init__()
        self._outfile = outfile
        self._append_mode = append_mode

        header = self._header()

        if Path(self._outfile).is_file() and append_mode:
            logger.info(f"Open file '{self._outfile}' for appending the results")
            _data_df = self.load_dataframe(self._outfile)
            loaded_header = _data_df.columns
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
            _data_df = pd.DataFrame(columns=header)
            _data_df.to_csv(self._outfile, index=False)

    def load_dataframe(self, filename):
        return pd.read_csv(filename)

    def _save_experiment(self, new_row: dict) -> bool:
        _data_df = self.load_dataframe(self._outfile)
        _data_df = _data_df.append(new_row, ignore_index=True)
        _data_df.to_csv(self._outfile, index=False)
        return True

    @abstractmethod
    def _header(self) -> Tuple[str]:
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

    def _get_geometry_header_width_infix_full(self, infix: str):
        return (f"mean_{infix}_reproj_error_col", f"mean_{infix}_reproj_error_row", f"mean_{infix}_reproj_error")

    def _construct_geometry_entries(self, infix: str, result: ReprojectionErrorResult) -> dict:
        return {
            f"mean_{infix}_reproj_error_col": result.mean_col_error,
            f"mean_{infix}_reproj_error_row": result.mean_row_error,
        }

    def _construct_geometry_entries_full(self, infix: str, result: ReprojectionErrorResult) -> dict:
        return {
            f"mean_{infix}_reproj_error_col": result.mean_col_error,
            f"mean_{infix}_reproj_error_row": result.mean_row_error,
            f"mean_{infix}_reproj_error": result.mean_error,
        }


class MPEvolutionResultWriter(MPResultWriter):
    def _header(self) -> Tuple[str]:
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
            noise_strategy: NoiseStrategy,
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
            "noise_type": noise_strategy.printable_identifier(),
            "noise_value": noise_strategy.get_value(),
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

    def save_experiments(
            self,
            strategy_bundle: StrategyBundle,
            distance_amount: Amount,
            results_start_camera: List[Camera],
            target_camera: Camera,
            results_result_camera: List[Camera],
            noise_strategy: NoiseStrategy,
            results_fitting_result: List[ReprojectionErrorResult],
            results_dense_result: List[ReprojectionErrorResult],
            results_y0_result: List[ReprojectionErrorResult],
            results_best_fitness: List[float],
            results_generations: List[int]
    ):
        _data_df = self.load_dataframe(self._outfile)

        for (start_camera, result_camera, fitting_result, dense_result, y0_result, best_fitness, generations) in zip(
                results_start_camera, results_result_camera, results_fitting_result, results_dense_result,
                results_y0_result, results_best_fitness, results_generations):
            new_row = {
                "population_fn": strategy_bundle.populate_strategy.printable_identifier(),
                "fitness_fn": strategy_bundle.fitness_strategy.printable_identifier(),
                "selection_fn": strategy_bundle.selection_strategy.printable_identifier(),
                "crossover_fn": strategy_bundle.crossover_strategy.printable_identifier(),
                "mutation_fn": strategy_bundle.mutation_strategy.printable_identifier(),
                "termination_fn": strategy_bundle.termination_strategy.printable_identifier(),
                "distance_type": distance_amount.name,
                "noise_type": noise_strategy.printable_identifier(),
                "noise_value": noise_strategy.get_value(),
                "best_fitness": best_fitness,
                "generations": generations,
            }

            new_row.update(self._construct_camera_entries("s", start_camera))
            new_row.update(self._construct_camera_entries("t", target_camera))
            new_row.update(self._construct_camera_entries("r", result_camera))

            new_row.update(self._construct_geometry_entries("fitting", fitting_result))
            new_row.update(self._construct_geometry_entries("dense", dense_result))
            new_row.update(self._construct_geometry_entries("y0", y0_result))
            _data_df = _data_df.append(new_row, ignore_index=True)

        _data_df.to_csv(self._outfile, index=False)

    def has(
            self,
            amount: Amount,
            strategy_bundle: StrategyBundle,
            noise_strategy: NoiseStrategy,
    ):
        _data_df = self.load_dataframe(self._outfile)
        population_filter = (
                _data_df["population_fn"] == strategy_bundle.populate_strategy.printable_identifier()
        )
        fitness_filter = _data_df["fitness_fn"] == strategy_bundle.fitness_strategy.printable_identifier()
        selection_filter = (
                _data_df["selection_fn"] == strategy_bundle.selection_strategy.printable_identifier()
        )
        crossover_filter = (
                _data_df["crossover_fn"] == strategy_bundle.crossover_strategy.printable_identifier()
        )
        mutation_filter = _data_df["mutation_fn"] == strategy_bundle.mutation_strategy.printable_identifier()
        noise_type_filter = _data_df["noise_type"] == noise_strategy.printable_identifier()
        noise_value_filter = _data_df["noise_value"] == noise_strategy.get_value()
        distance_type_filter = _data_df["distance_type"] == amount.name

        return len(
            self.load_dataframe(self._outfile)[
                population_filter
                & fitness_filter
                & selection_filter
                & crossover_filter
                & mutation_filter
                & noise_type_filter
                & noise_value_filter
                & distance_type_filter
                ]
        )


class MPNloptResultWriter(MPResultWriter):
    def _header(self) -> Tuple[str]:
        # region [Region0] Nlopt Specification + Noise Header
        nlopt_type = ("nlopt_optimizer",
                      "fitness_fn",
                      "distance_type")

        noise_columns = ("noise_type", "noise_value")
        # endregion

        # region [Region1] Start, Target and Result Camera Header
        start_camera_header = self._get_camera_header_with_prefix("s")
        target_camera_header = self._get_camera_header_with_prefix("t")
        result_camera_header = self._get_camera_header_with_prefix("r")
        # endregion

        # region [Region2] Fitness + Generation Header
        results = ("best_fitness", "duration_in_s",)
        # endregion

        # region [Region3] Result for various geometries
        fitting_geometry_result = self._get_geometry_header_width_infix_full("fitting")
        dense_geometry_result = self._get_geometry_header_width_infix_full("dense")
        y0_geometry_result = self._get_geometry_header_width_infix_full("y0")
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

    def save_experiments(
            self,
            algorithm_name: str,
            fitness_strategy: FitnessStrategy,
            distance_amount: Amount,
            results_start_camera: List[Camera],
            target_camera: Camera,
            results_result_camera: List[Camera],
            noise_strategy: NoiseStrategy,
            results_fitting_result: List[ReprojectionErrorResult],
            results_dense_result: List[ReprojectionErrorResult],
            results_y0_result: List[ReprojectionErrorResult],
            results_best_fitness: List[float],
            results_duration_in_s: List[float]
    ):
        _data_df = self.load_dataframe(self._outfile)

        for (start_camera, result_camera, fitting_result, dense_result, y0_result, best_fitness, duration_in_s) in zip(
                results_start_camera, results_result_camera, results_fitting_result, results_dense_result,
                results_y0_result, results_best_fitness, results_duration_in_s):
            new_row = {
                "nlopt_optimizer": algorithm_name,
                "fitness_fn": fitness_strategy.printable_identifier(),
                "distance_type": distance_amount.name,
                "noise_type": noise_strategy.printable_identifier(),
                "noise_value": noise_strategy.get_value(),
                "best_fitness": best_fitness,
                "duration_in_s": duration_in_s
            }

            new_row.update(self._construct_camera_entries("s", start_camera))
            new_row.update(self._construct_camera_entries("t", target_camera))
            new_row.update(self._construct_camera_entries("r", result_camera))

            new_row.update(self._construct_geometry_entries_full("fitting", fitting_result))
            new_row.update(self._construct_geometry_entries_full("dense", dense_result))
            new_row.update(self._construct_geometry_entries_full("y0", y0_result))
            _data_df = _data_df.append(new_row, ignore_index=True)

        _data_df.to_csv(self._outfile, index=False)

    def save_experiment(
            self,
            nlopt_optimizer: str,
            fitness_strategy: FitnessStrategy,
            distance_amount: Amount,
            start_camera: Camera,
            target_camera: Camera,
            result_camera: Camera,
            noise_strategy: NoiseStrategy,
            fitting_result: ReprojectionErrorResult,
            dense_result: ReprojectionErrorResult,
            y0_result: ReprojectionErrorResult,
            best_fitness: float,
            duration_in_s: float
    ):
        new_row = {
            "nlopt_optimizer": nlopt_optimizer,
            "fitness_fn": fitness_strategy.printable_identifier(),
            "distance_type": distance_amount.name,
            "noise_type": noise_strategy.printable_identifier(),
            "noise_value": noise_strategy.get_value(),
            "best_fitness": best_fitness,
            "duration_in_s": duration_in_s
        }

        new_row.update(self._construct_camera_entries("s", start_camera))
        new_row.update(self._construct_camera_entries("t", target_camera))
        new_row.update(self._construct_camera_entries("r", result_camera))

        new_row.update(self._construct_geometry_entries("fitting", fitting_result))
        new_row.update(self._construct_geometry_entries("dense", dense_result))
        new_row.update(self._construct_geometry_entries("y0", y0_result))

        self._save_experiment(new_row)

    def has(self, amount: Amount, optimizer: NloptAlgorithm, fitness_strategy: FitnessStrategy,
            noise_strategy: NoiseStrategy):
        _data_df = self.load_dataframe(self._outfile)

        distance_type_filter = _data_df["distance_type"] == amount.name
        nlopt_optimizer_filter = _data_df["nlopt_optimizer"] == optimizer.display_name

        fitness_filter = _data_df["fitness_fn"] == fitness_strategy.printable_identifier()

        noise_type_filter = _data_df["noise_type"] == noise_strategy.printable_identifier()
        noise_value_filter = _data_df["noise_value"] == noise_strategy.get_value()
        return len(
            self.load_dataframe(self._outfile)[
                distance_type_filter
                & fitness_filter
                & nlopt_optimizer_filter
                & noise_type_filter
                & noise_value_filter])
