import numpy as np
from scipy.spatial import Delaunay as ScipyDelaunay

from .mesh import Mesh
from .utils import compute_normals, compute_uvs_planar


class DelaunayTriangulator:
    def triangulate(
        self,
        points: np.ndarray,
        heights: np.ndarray | None = None,
        add_bounding_box: bool = False,
        z_scale: float = 1.0,
        flat: bool = False,
    ) -> Mesh:
        N = points.shape[0]
        if N < 3:
            return Mesh(
                vertices=np.empty((0, 3), dtype=np.float32),
                indices=np.empty((0, 3), dtype=np.uint32),
                normals=np.empty((0, 3), dtype=np.float32),
                uvs=np.empty((0, 2), dtype=np.float32),
            )

        if add_bounding_box:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            corners = np.array(
                [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]],
                dtype=np.float32,
            )
            points = np.vstack([points, corners])
            N = points.shape[0]
            if heights is not None:
                heights = np.append(heights, [0.0, 0.0, 0.0, 0.0])

        tri = ScipyDelaunay(points)
        if heights is None:
            heights = np.zeros(N, dtype=np.float32)

        heights = heights * z_scale

        vertices = np.column_stack(
            [
                points[:, 0].astype(np.float32),
                heights.astype(np.float32),
                points[:, 1].astype(np.float32),
            ]
        )
        indices = tri.simplices.astype(np.uint32)
        normals = compute_normals(vertices, indices)
        uvs = compute_uvs_planar(vertices)
        if flat:
            return self._make_flat(vertices, indices, normals, uvs)
        return Mesh(
            vertices=vertices, indices=indices, normals=normals, uvs=uvs
        )

    @staticmethod
    def _make_flat(
        vertices: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray,
        uvs: np.ndarray,
    ) -> Mesh:
        M = indices.shape[0]
        flat_vertices = vertices[indices.ravel()].copy()
        flat_uvs = uvs[indices.ravel()].copy()
        flat_indices = np.arange(M * 3, dtype=np.uint32).reshape(-1, 3)

        v0 = flat_vertices[flat_indices[:, 0]]
        v1 = flat_vertices[flat_indices[:, 1]]
        v2 = flat_vertices[flat_indices[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        fnorms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        fnorms[fnorms == 0] = 1
        face_normals /= fnorms

        flat_normals = np.repeat(face_normals, 3, axis=0)

        return Mesh(
            vertices=flat_vertices,
            indices=flat_indices,
            normals=flat_normals,
            uvs=flat_uvs,
        )

    def triangulate_with_constraints(
        self,
        points: np.ndarray,
        heights: np.ndarray | None = None,
        boundary: np.ndarray | None = None,
    ) -> Mesh:
        return self.triangulate(points, heights)
