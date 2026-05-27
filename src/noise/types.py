from dataclasses import dataclass
import numpy as np

@dataclass
class NoiseMap:
    data: np.ndarray
    width: int
    height: int
    seed: int

@dataclass
class PointSet:
    points: np.ndarray
    width: float
    height: float
    seed: int