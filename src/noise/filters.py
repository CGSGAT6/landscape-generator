import numpy as np
from scipy.interpolate import interp1d
from .types import NoiseMap


class NoiseFilter:
    @staticmethod
    def power_curve(noise_map: NoiseMap, exponent: float) -> NoiseMap:
        data = np.power(noise_map.data, exponent)
        return NoiseMap(data=data.astype(np.float32),
                        width=noise_map.width,
                        height=noise_map.height,
                        seed=noise_map.seed)

    @staticmethod
    def step(noise_map: NoiseMap, levels: int) -> NoiseMap:
        data = np.floor(noise_map.data * levels) / (levels - 1)
        data = np.clip(data, 0.0, 1.0)
        return NoiseMap(data=data.astype(np.float32),
                        width=noise_map.width,
                        height=noise_map.height,
                        seed=noise_map.seed)

    @staticmethod
    def remap(noise_map: NoiseMap,
              in_min: float, in_max: float,
              out_min: float, out_max: float) -> NoiseMap:
        data = (noise_map.data - in_min) / (in_max - in_min)
        data = data * (out_max - out_min) + out_min
        data = np.clip(data, 0.0, 1.0)
        return NoiseMap(data=data.astype(np.float32),
                        width=noise_map.width,
                        height=noise_map.height,
                        seed=noise_map.seed)

    @staticmethod
    def apply_curve(noise_map: NoiseMap,
                    control_points: list[tuple[float, float]]) -> NoiseMap:
        x = np.array([p[0] for p in control_points])
        y = np.array([p[1] for p in control_points])
        curve = interp1d(x, y, kind='cubic', fill_value='extrapolate')
        data = curve(noise_map.data)
        data = np.clip(data, 0.0, 1.0)
        return NoiseMap(data=data.astype(np.float32),
                        width=noise_map.width,
                        height=noise_map.height,
                        seed=noise_map.seed)