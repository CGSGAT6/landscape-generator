import numpy as np


def compute_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0 or len(vertices) == 0:
        return np.zeros_like(vertices)

    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    face_normals /= norms

    vertex_normals = np.zeros_like(vertices)
    np.add.at(vertex_normals, indices[:, 0], face_normals)
    np.add.at(vertex_normals, indices[:, 1], face_normals)
    np.add.at(vertex_normals, indices[:, 2], face_normals)

    vnorms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vnorms[vnorms == 0] = 1
    vertex_normals /= vnorms
    return vertex_normals


def compute_uvs_planar(vertices: np.ndarray) -> np.ndarray:
    if len(vertices) == 0:
        return np.empty((0, 2), dtype=np.float32)

    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    x_range = x_max - x_min
    z_range = z_max - z_min
    if x_range == 0:
        x_range = 1
    if z_range == 0:
        z_range = 1

    u = (vertices[:, 0] - x_min) / x_range
    v = (vertices[:, 2] - z_min) / z_range
    return np.column_stack([u, v])


def merge_meshes(meshes: list) -> "Mesh":
    from .mesh import Mesh

    if not meshes:
        return Mesh(
            vertices=np.empty((0, 3), dtype=np.float32),
            indices=np.empty((0, 3), dtype=np.uint32),
            normals=np.empty((0, 3), dtype=np.float32),
            uvs=np.empty((0, 2), dtype=np.float32),
        )

    vertices = np.concatenate([m.vertices for m in meshes])
    normals = np.concatenate([m.normals for m in meshes])
    uvs = np.concatenate([m.uvs for m in meshes])

    offset = 0
    indices_list = []
    for m in meshes:
        indices_list.append(m.indices + offset)
        offset += m.vertex_count
    indices = np.concatenate(indices_list)

    return Mesh(vertices=vertices, indices=indices, normals=normals, uvs=uvs)
