from dataclasses import dataclass
from typing import Union, List

import numpy as np
from evolution.base import BaseGeometry
from evolution.camera.camera_rendering import project_points

from cameras.cameras import Camera
from utils.transform_utils import split_dna


@dataclass
class ReprojectionErrorResult:
    """Dataclass for reprojection error related information"""

    """ Projected image points from camera (A) """
    projected_points_a: np.array
    """ Projected image points from camera (B) """
    projected_points_b: np.array
    """ Distances for projected_points_b - projected_points_a """
    point_distances: np.array

    """ Point individual errors sqrt(sum(point_distances**2))"""
    point_errors: np.array
    """ Mean (euclidean distance) reprojection error """
    mean_error: float
    """ Mean (column) reprojection error """
    mean_col_error: float
    """ Mean (row) reprojection error """
    mean_row_error: float


def reprojection_error(
        camera_a: Union[Camera, np.array], camera_b: Union[Camera, np.array], geometry: BaseGeometry
) -> ReprojectionErrorResult:
    cma, ta, ra, da = split_dna(camera_a.dna if type(camera_a) == Camera else camera_a)
    cmb, tb, rb, db = split_dna(camera_b.dna if type(camera_b) == Camera else camera_b)
    _projected_points_a = project_points(geometry.world_points, cma, ta, ra, da)
    _projected_points_b = project_points(geometry.world_points, cmb, tb, rb, db)

    _projected_points_a = _projected_points_a.reshape((-1, 2))
    _projected_points_b = _projected_points_b.reshape((-1, 2))
    _point_distances = _projected_points_b - _projected_points_a
    _point_errors = np.sqrt(np.sum(_point_distances ** 2, axis=1))
    _mean_error = np.mean(_point_errors)

    _mean_col_err = np.mean(abs(_point_distances[:, 0]))
    _mean_row_err = np.mean(abs(_point_distances[:, 1]))

    return ReprojectionErrorResult(
        _projected_points_a,
        _projected_points_b,
        _point_distances,
        _point_errors,
        _mean_error,
        _mean_col_err,
        _mean_row_err,
    )


def reprojection_error_multiple_geometries(
        camera_a: Union[Camera, np.array],
        camera_b: Union[Camera, np.array],
        geometries: List[BaseGeometry],
) -> list[ReprojectionErrorResult]:
    return [reprojection_error(camera_a, camera_b, geometry) for geometry in geometries]
