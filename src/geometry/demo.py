import numpy as np
from pathlib import Path

from geometry import GridBuilder, DelaunayTriangulator, save_obj, load_obj


def demo():
    print("=== geometry module demo ===\n")

    hmap = np.random.default_rng(42).random((32, 48), dtype=np.float32)

    builder = GridBuilder()
    grid_mesh = builder.build(hmap, x_scale=0.5, y_scale=0.5, z_scale=15.0)
    print(f"Grid mesh: {grid_mesh.vertex_count} vertices, {grid_mesh.face_count} faces")

    rng = np.random.default_rng(123)
    pts = rng.uniform(0, 48, (200, 2)).astype(np.float32)
    hs = rng.uniform(0, 1, (200,)).astype(np.float32)

    delaunay = DelaunayTriangulator()
    tri_mesh = delaunay.triangulate(pts, hs, add_bounding_box=True, z_scale=15.0, flat=True)
    print(f"Delaunay mesh (+bounding box): {tri_mesh.vertex_count} vertices, {tri_mesh.face_count} faces")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    obj_path = out_dir / "terrain_grid.obj"
    save_obj(grid_mesh, str(obj_path))
    print(f"Saved OBJ: {obj_path}")

    obj_path2 = out_dir / "terrain_delaunay.obj"
    save_obj(tri_mesh, str(obj_path2))
    print(f"Saved OBJ: {obj_path2}")

    loaded = load_obj(str(obj_path))
    print(f"Loaded OBJ: {loaded.vertex_count} vertices, {loaded.face_count} faces")

    print("\n=== demo completed ===")


if __name__ == "__main__":
    demo()
