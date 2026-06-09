from pathlib import Path

import numpy as np

from filepath import OUTPUT_DIR
from .landscape import Landscape
from .generate import generate_landscape


def demo():
    land = generate_landscape(
        width=64, height=64,
        seed_height=47, seed_moisture=18, seed_trees=30,
        sea_level=0.2,
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
    assert loaded.grid_mesh.vertex_count == land.grid_mesh.vertex_count, "grid vertex count"
    assert loaded.grid_mesh.face_count == land.grid_mesh.face_count, "grid face count"
    assert loaded.flat_mesh.face_count == land.flat_mesh.face_count, "flat face count"
    print("Roundtrip OK")


if __name__ == "__main__":
    demo()
