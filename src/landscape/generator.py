from __future__ import annotations

import numpy as np
from PIL import Image

from filepath import OUTPUT_DIR
from geometry import GridBuilder, DelaunayTriangulator
from noise import FractalNoise, PerlinNoise, PoissonSampler, NoiseFilter
from .landscape import Landscape
from .biome import BiomeType


_BIOME_COLORS = {
    BiomeType.WATER:  (30, 100, 200),
    BiomeType.SAND:   (240, 210, 160),
    BiomeType.GRASS:  (80, 160, 60),
    BiomeType.FOREST: (30, 100, 30),
    BiomeType.ROCK:   (130, 120, 110),
    BiomeType.SNOW:   (230, 235, 240),
}


def _make_biome_texture(biome_map: np.ndarray) -> Image.Image:
    H, W = biome_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for btype, color in _BIOME_COLORS.items():
        mask = biome_map == btype
        rgb[mask] = color
    return Image.fromarray(rgb, mode="RGB")


def _sample_height_at(points: np.ndarray,
                      height_map: np.ndarray,
                      x_scale: float,
                      y_scale: float,
                      z_scale: float) -> np.ndarray:
    col = (points[:, 0] / x_scale).astype(int)
    row = (points[:, 1] / y_scale).astype(int)
    col = np.clip(col, 0, height_map.shape[1] - 1)
    row = np.clip(row, 0, height_map.shape[0] - 1)
    return height_map[row, col] * z_scale


class Generator:
    def __init__(self):
        self._noise_cache: dict[int, PerlinNoise] = {}
        self._landscape: Landscape | None = None

    def _get_noise(self, seed: int) -> PerlinNoise:
        noise = self._noise_cache.get(seed)
        if noise is None:
            noise = PerlinNoise(seed)
            self._noise_cache[seed] = noise
        return noise

    @property
    def landscape(self) -> Landscape | None:
        return self._landscape

    def generate(
        self,
        width: int = 256,
        height: int = 256,
        seed_height: int = 47,
        seed_moisture: int = 18,
        seed_trees: int = 30,
        sea_level: float = 0.2,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 20.0,
        detail: float = 50.0,
        noise_decay: float = 0.5,
        frequency: float = 0.05,
        lacunarity: float = 2.0,
        power: float = 1.0,
        tree_density: float = 30.0,
    ) -> Landscape:
        octaves = max(1, int(round(detail / 10)))
        gain = noise_decay
        scale = frequency
        exponent = power
        min_dist = max(0.5, 5.0 - tree_density * 0.045)

        base = self._get_noise(seed_height)
        noise = FractalNoise(base)
        nm = noise.fbm(width, height, octaves=octaves, gain=gain, scale=scale, lacunarity=lacunarity)
        height_map = NoiseFilter.power_curve(nm, exponent).data

        biome_map = np.zeros((height, width), dtype=np.uint8)
        biome_map[height_map < sea_level] = BiomeType.WATER
        biome_map[(height_map >= sea_level) & (height_map < 0.35)] = BiomeType.SAND
        biome_map[(height_map >= 0.35) & (height_map < 0.6)] = BiomeType.GRASS
        biome_map[(height_map >= 0.6) & (height_map < 0.8)] = BiomeType.FOREST
        biome_map[height_map >= 0.8] = BiomeType.ROCK

        grid_mesh = GridBuilder().build(height_map,
                                         x_scale=x_scale,
                                         y_scale=y_scale,
                                         z_scale=z_scale)

        sampler = PoissonSampler(seed_trees)
        pts = sampler.sample(width * x_scale, height * y_scale, min_distance=min_dist)
        heights = _sample_height_at(pts.points, height_map,
                                    x_scale, y_scale, z_scale)
        flat_mesh = DelaunayTriangulator().triangulate(
            pts.points, heights=heights,
            z_scale=1.0, flat=True, add_bounding_box=True,
        )

        tree_positions = np.array([
            [5.0, height_map[10, 10] * z_scale, 5.0],
            [15.0, height_map[20, 30] * z_scale, 15.0],
            [25.0, height_map[40, 20] * z_scale, 25.0],
        ], dtype=np.float32)

        tex_path = OUTPUT_DIR / "landscape_tex.png"
        tex_img = _make_biome_texture(biome_map)
        tex_img.save(tex_path)

        self._landscape = Landscape(
            height_map=height_map,
            biome_map=biome_map,
            texture_path=tex_path,
            texture_image=tex_img,
            grid_mesh=grid_mesh,
            flat_mesh=flat_mesh,
            sea_level=sea_level,
            tree_positions=tree_positions,
            seed_height=seed_height,
            seed_moisture=seed_moisture,
            seed_trees=seed_trees,
        )
        return self._landscape
