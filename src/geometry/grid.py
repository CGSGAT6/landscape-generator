import numpy as np

from .mesh import Mesh
from .utils import compute_normals


class GridBuilder:
    def build(
        self,
        heightmap: np.ndarray,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 5.0,
    ) -> Mesh:
        H, W = heightmap.shape

        xs = np.arange(W, dtype=np.float32) * x_scale
        zs = np.arange(H, dtype=np.float32) * y_scale
        xx, zz = np.meshgrid(xs, zs)
        vertices = np.stack(
            [
                xx.ravel(),
                (heightmap.ravel() * z_scale).astype(np.float32),
                zz.ravel(),
            ],
            axis=1,
        )

        u = np.tile(np.linspace(0, 1, W, dtype=np.float32), H)
        v = np.repeat(np.linspace(0, 1, H, dtype=np.float32), W)
        uvs = np.column_stack([u, v])

        if H < 2 or W < 2:
            indices = np.empty((0, 3), dtype=np.uint32)
        else:
            i_idx = np.arange(H - 1)
            j_idx = np.arange(W - 1)
            ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")
            ii = ii.ravel()
            jj = jj.ravel()

            tl = ii * W + jj
            tr = ii * W + jj + 1
            bl = (ii + 1) * W + jj
            br = (ii + 1) * W + jj + 1

            indices = np.vstack(
                [
                    np.column_stack([tl, bl, tr]),
                    np.column_stack([tr, bl, br]),
                ]
            ).astype(np.uint32)

        normals = compute_normals(vertices, indices)
        return Mesh(
            vertices=vertices, indices=indices, normals=normals, uvs=uvs
        )

    def build_with_lod(
        self, heightmap: np.ndarray, levels: int = 3, **kwargs
    ) -> list[Mesh]:
        meshes = []
        for level in range(levels):
            step = 2**level
            sub = heightmap[::step, ::step]
            meshes.append(self.build(sub, **kwargs))
        return meshes
