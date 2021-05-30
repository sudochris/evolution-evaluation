from typing import Tuple

import numpy as np


def split_dna(camera_dna: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
    x = camera_dna
    camera_matrix = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1]], dtype=np.float32)

    t_vec = x[4:7].astype(np.float32)
    r_vec = x[7:10].astype(np.float32)
    d_vec = x[10:].astype(np.float32)

    return camera_matrix, t_vec, r_vec, d_vec
