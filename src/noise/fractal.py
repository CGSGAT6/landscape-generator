import numpy as np
from .perlin import PerlinNoise
from .types import NoiseMap


class FractalNoise:
    def __init__(self, base_noise: PerlinNoise):
        self.base = base_noise

    def fbm(self, width: int, height: int,
            octaves: int = 6,
            lacunarity: float = 2.0,
            gain: float = 0.5,
            scale: float = 0.05) -> NoiseMap:
        data = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        frequency = scale
        max_value = 0.0

        for _ in range(octaves):
            octave_map = self.base.generate2d(width, height, scale=frequency)
            data += octave_map.data * amplitude
            max_value += amplitude
            amplitude *= gain
            frequency *= lacunarity

        data /= max_value
        return NoiseMap(data=data, width=width, height=height, seed=0)

    def ridged(self, width: int, height: int,
               octaves: int = 6,
               lacunarity: float = 2.0,
               gain: float = 0.5,
               scale: float = 1.0) -> NoiseMap:
        data = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        frequency = scale
        max_value = 0.0

        for _ in range(octaves):
            octave_map = self.base.generate2d(width, height, scale=frequency)
            octave_data = 1.0 - np.abs(octave_map.data * 2.0 - 1.0)
            data += octave_data * amplitude
            max_value += amplitude
            amplitude *= gain
            frequency *= lacunarity

        data /= max_value
        return NoiseMap(data=data, width=width, height=height, seed=0)

    def red_noise(self, width: int, height: int,
                  exponent: float = 2.0) -> NoiseMap:
        white = np.random.randn(height, width).astype(np.float32)
        fft = np.fft.fft2(white)
        freq_x = np.fft.fftfreq(width)
        freq_y = np.fft.fftfreq(height)
        fx, fy = np.meshgrid(freq_x, freq_y)
        f = np.sqrt(fx**2 + fy**2)
        f[0, 0] = 1.0
        fft *= 1.0 / (f ** (exponent / 2.0))
        data = np.real(np.fft.ifft2(fft))
        data = (data - data.min()) / (data.max() - data.min())
        return NoiseMap(data=data.astype(np.float32), width=width, height=height, seed=0)