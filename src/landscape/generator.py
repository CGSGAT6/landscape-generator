from __future__ import annotations

import numpy as np
from PIL import Image

from filepath import OUTPUT_DIR
from geometry import GridBuilder, DelaunayTriangulator
from noise import FractalNoise, PerlinNoise, PoissonSampler, NoiseFilter
from .landscape import Landscape
from .biome import BiomeType
from .selector import Selector, apply_selector_to_arrays, _make_default_selector


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


# ─── Применение Selector к массивам (импортируется из selector.py) ──


# ─── Непрерывная генерация текстуры ──────────────────────────────────


def _sample_fbm_grid(x_grid: np.ndarray,
                     y_grid: np.ndarray,
                     base_noise: PerlinNoise,
                     octaves: int, gain: float,
                     scale: float, lacunarity: float) -> np.ndarray:
    data = np.zeros_like(x_grid, dtype=np.float32)
    amplitude = 1.0
    freq = scale
    max_value = 0.0
    for _ in range(octaves):
        data += base_noise._evaluate_grid(x_grid * freq, y_grid * freq) * amplitude
        max_value += amplitude
        amplitude *= gain
        freq *= lacunarity
    data /= max_value
    return data


def _generate_continuous_texture(
    tex_w: int, tex_h: int,
    grid_w: int, grid_h: int,
    base_h: PerlinNoise, base_m: PerlinNoise,
    octaves: int, gain: float, scale: float, lacunarity: float,
    exponent: float,
    selector: Selector,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    px = np.linspace(0, grid_w, tex_w, endpoint=False)
    py = np.linspace(0, grid_h, tex_h, endpoint=False)
    x_grid, y_grid = np.meshgrid(px, py)

    height_raw = _sample_fbm_grid(x_grid, y_grid, base_h,
                                   octaves, gain, scale, lacunarity)
    height_raw = np.clip(height_raw, 0.0, 1.0)
    height_vals = np.power(height_raw, exponent)

    moisture_vals = _sample_fbm_grid(x_grid, y_grid, base_m,
                                      octaves, gain, scale, lacunarity)
    moisture_vals = np.clip(moisture_vals, 0.0, 1.0)

    biome_map = apply_selector_to_arrays(selector, height_vals, moisture_vals)

    rgb = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)
    for btype, color in _BIOME_COLORS.items():
        mask = biome_map == btype
        rgb[mask] = color
    biome_img = Image.fromarray(rgb, mode="RGB")

    height_img = Image.fromarray(
        (height_vals * 255).astype(np.uint8), mode="L"
    )
    moisture_img = Image.fromarray(
        (moisture_vals * 255).astype(np.uint8), mode="L"
    )
    return biome_img, height_img, moisture_img


# ─── Generator ────────────────────────────────────────────────────────


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
        detail: float = 3.0,
        noise_decay: float = 0.5,
        frequency: float = 0.05,
        lacunarity: float = 2.0,
        power: float = 1.0,
        tree_density: float = 30.0,
        texture_resolution: int = 1,
        selector: Selector | None = None,
    ) -> Landscape:
        octaves = 6
        gain = noise_decay
        scale = frequency
        exponent = power
        min_dist = max(0.5, 5.0 - tree_density * 0.045)

        grid_detail = max(1, int(round(detail)))
        grid_w = max(2, int(round(width * grid_detail)))
        grid_h = max(2, int(round(height * grid_detail)))
        cell_w = 1.0 / grid_detail
        cell_h = 1.0 / grid_detail

        if selector is None:
            selector = _make_default_selector(sea_level)

        base_h = self._get_noise(seed_height)
        noise_h = FractalNoise(base_h)
        hm_noise = noise_h.fbm(grid_w, grid_h, octaves=octaves,
                                gain=gain, scale=scale, lacunarity=lacunarity)
        height_map = NoiseFilter.power_curve(hm_noise, exponent).data

        base_m = self._get_noise(seed_moisture)
        noise_m = FractalNoise(base_m)
        mm_noise = noise_m.fbm(grid_w, grid_h, octaves=octaves,
                                gain=gain, scale=scale, lacunarity=lacunarity)
        moisture_map = np.clip(mm_noise.data, 0.0, 1.0)

        biome_map = apply_selector_to_arrays(selector, height_map, moisture_map)

        grid_mesh = GridBuilder().build(height_map,
                                         x_scale=cell_w,
                                         y_scale=cell_h,
                                         z_scale=z_scale)

        sampler = PoissonSampler(seed_trees)
        pts = sampler.sample(width, height, min_distance=min_dist)
        heights = _sample_height_at(pts.points, height_map,
                                    cell_w, cell_h, z_scale)
        flat_mesh = DelaunayTriangulator().triangulate(
            pts.points, heights=heights,
            z_scale=1.0, flat=True, add_bounding_box=True,
        )

        tree_positions = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                                  dtype=np.float32)

        tex_res = max(1, texture_resolution)
        tex_w = grid_w * tex_res
        tex_h = grid_h * tex_res
        tex_img, height_img, moisture_img = _generate_continuous_texture(
            tex_w, tex_h,
            grid_w, grid_h,
            base_h, base_m,
            octaves, gain, scale, lacunarity, exponent,
            selector,
        )
        tex_path = OUTPUT_DIR / "landscape_tex.png"
        tex_img.save(tex_path)
        height_img.save(OUTPUT_DIR / "landscape_height.png")
        moisture_img.save(OUTPUT_DIR / "landscape_moisture.png")

        self._landscape = Landscape(
            height_map=height_map,
            biome_map=biome_map,
            moisture_map=moisture_map,
            texture_path=tex_path,
            texture_image=tex_img,
            height_image=height_img,
            moisture_image=moisture_img,
            grid_mesh=grid_mesh,
            flat_mesh=flat_mesh,
            sea_level=sea_level,
            tree_positions=tree_positions,
            seed_height=seed_height,
            seed_moisture=seed_moisture,
            seed_trees=seed_trees,
        )
        return self._landscape
