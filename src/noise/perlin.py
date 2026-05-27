import numpy as np
from .types import NoiseMap

class PerlinNoise:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256)
        self.perm = np.concatenate([p, p])

    def _fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def noise1d(self, x: float) -> float:
        x0 = int(np.floor(x)) & 255
        x1 = (x0 + 1) & 255
        t = x - np.floor(x)
        u = self._fade(t)
        v0 = self.perm[x0] / 255.0
        v1 = self.perm[x1] / 255.0
        return v0 + (v1 - v0) * u

    def noise2d(self, x: float, y: float) -> float:
        x0 = int(np.floor(x)) & 255
        y0 = int(np.floor(y)) & 255
        x1 = (x0 + 1) & 255
        y1 = (y0 + 1) & 255
        tx = x - np.floor(x)
        ty = y - np.floor(y)
        u = self._fade(tx)
        v = self._fade(ty)
        p = self.perm
        v00 = p[p[x0] + y0] / 255.0
        v10 = p[p[x1] + y0] / 255.0
        v01 = p[p[x0] + y1] / 255.0
        v11 = p[p[x1] + y1] / 255.0
        a = v00 + (v10 - v00) * u
        b = v01 + (v11 - v01) * u
        return a + (b - a) * v

    def noise3d(self, x: float, y: float, z: float) -> float:
        x0 = int(np.floor(x)) & 255
        y0 = int(np.floor(y)) & 255
        z0 = int(np.floor(z)) & 255
        x1 = (x0 + 1) & 255
        y1 = (y0 + 1) & 255
        z1 = (z0 + 1) & 255
        tx = x - np.floor(x)
        ty = y - np.floor(y)
        tz = z - np.floor(z)
        u = self._fade(tx)
        v = self._fade(ty)
        w = self._fade(tz)
        p = self.perm
        v000 = p[p[p[x0] + y0] + z0] / 255.0
        v100 = p[p[p[x1] + y0] + z0] / 255.0
        v010 = p[p[p[x0] + y1] + z0] / 255.0
        v110 = p[p[p[x1] + y1] + z0] / 255.0
        v001 = p[p[p[x0] + y0] + z1] / 255.0
        v101 = p[p[p[x1] + y0] + z1] / 255.0
        v011 = p[p[p[x0] + y1] + z1] / 255.0
        v111 = p[p[p[x1] + y1] + z1] / 255.0
        a = v000 + (v100 - v000) * u
        b = v010 + (v110 - v010) * u
        c = v001 + (v101 - v001) * u
        d = v011 + (v111 - v011) * u
        e = a + (b - a) * v
        f = c + (d - c) * v
        return e + (f - e) * w

    def generate2d(self, width: int, height: int, scale: float = 1.0) -> NoiseMap:
        x = np.linspace(0, width * scale, width, endpoint=False)
        y = np.linspace(0, height * scale, height, endpoint=False)
        data = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                data[i, j] = self.noise2d(x[j], y[i])
        return NoiseMap(data=data, width=width, height=height, seed=0)