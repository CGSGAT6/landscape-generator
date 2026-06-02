import numpy as np
from pathlib import Path

from filepath import OUTPUT_DIR
from geometry import GridBuilder, DelaunayTriangulator
from noise import FractalNoise, PerlinNoise, PoissonSampler
from .landscape import Landscape
from .biome import BiomeType


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


def demo():
    W, H = 64, 64
    seed_h, seed_m, seed_t = 42, 100, 200
    x_scale, y_scale, z_scale = 0.5, 0.5, 20.0

    noise = FractalNoise(PerlinNoise(seed_h))
    height_map = noise.fbm(W, H, octaves=6, scale=0.05).data

    biome_map = np.zeros((H, W), dtype=np.uint8)
    biome_map[height_map < 0.2] = BiomeType.WATER
    biome_map[(height_map >= 0.2) & (height_map < 0.35)] = BiomeType.SAND
    biome_map[(height_map >= 0.35) & (height_map < 0.6)] = BiomeType.GRASS
    biome_map[(height_map >= 0.6) & (height_map < 0.8)] = BiomeType.FOREST
    biome_map[height_map >= 0.8] = BiomeType.ROCK

    grid_mesh = GridBuilder().build(height_map,
                                     x_scale=x_scale,
                                     y_scale=y_scale,
                                     z_scale=z_scale)

    sampler = PoissonSampler(seed_t)
    pts = sampler.sample(W * x_scale, H * y_scale, min_distance=3.0)
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

    land = Landscape(
        height_map=height_map,
        biome_map=biome_map,
        grid_mesh=grid_mesh,
        flat_mesh=flat_mesh,
        sea_level=0.2,
        tree_positions=tree_positions,
        seed_height=seed_h,
        seed_moisture=seed_m,
        seed_trees=seed_t,
    )

    out = OUTPUT_DIR / "demo_landscape"
    land.save(out)
    print(f"Saved to {out.resolve()}")

    loaded = Landscape.load(out)
    assert np.allclose(land.height_map, loaded.height_map), "height mismatch"
    assert np.array_equal(land.biome_map, loaded.biome_map), "biome mismatch"
    assert land.sea_level == loaded.sea_level, "sea_level mismatch"
    assert land.seed_height == loaded.seed_height
    assert land.seed_moisture == loaded.seed_moisture
    assert land.seed_trees == loaded.seed_trees
    assert loaded.grid_mesh is not None, "grid_mesh not loaded"
    assert loaded.flat_mesh is not None, "flat_mesh not loaded"
    assert loaded.tree_positions is not None, "tree_positions not loaded"
    atol = 1e-5
    assert np.allclose(land.grid_mesh.vertices, loaded.grid_mesh.vertices, atol=atol), "grid mesh mismatch"
    assert np.allclose(land.flat_mesh.vertices, loaded.flat_mesh.vertices, atol=atol), "flat mesh mismatch"
    print("Roundtrip OK")


if __name__ == "__main__":
    demo()
