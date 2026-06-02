from pathlib import Path

import numpy as np

from .mesh import Mesh


def save_obj(mesh: Mesh, path: str | Path,
             material_name: str | None = None,
             mtl_filename: str | None = None) -> None:
    lines = ["# Exported by landscape-generator geometry module\n"]
    if mtl_filename:
        lines.append(f"mtllib {mtl_filename}\n")

    for v in mesh.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for vt in mesh.uvs:
        lines.append(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
    for vn in mesh.normals:
        lines.append(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

    if material_name:
        lines.append(f"usemtl {material_name}\n")
    for f in mesh.indices:
        i0, i1, i2 = f[0] + 1, f[1] + 1, f[2] + 1
        lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

    Path(path).write_text("".join(lines))


def save_mtl(path: str | Path, material_name: str,
             texture_rel_path: str | None = None) -> None:
    lines = [
        f"newmtl {material_name}\n",
        "Ka 0.47 0.30 0.18\n",
        "Kd 0.47 0.30 0.18\n",
        "Ks 0.47 0.30 0.18\n",
        "Ns 500.0\n",
        "d 1.0\n",
        "illum 2\n",
    ]
    if texture_rel_path:
        lines.append(f"map_Kd {texture_rel_path}\n")
    Path(path).write_text("".join(lines))


def load_obj(path: str) -> Mesh:
    positions = []
    normals = []
    uvs = []
    faces = []

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "vn":
            normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "vt":
            uvs.append([float(parts[1]), float(parts[2])])
        elif parts[0] == "f":
            verts = []
            for p in parts[1:]:
                idx = p.split("/")[0]
                verts.append(int(idx) - 1)
            for i in range(1, len(verts) - 1):
                faces.append([verts[0], verts[i], verts[i + 1]])

    vertices_arr = np.array(positions, dtype=np.float32)
    indices_arr = np.array(faces, dtype=np.uint32)

    if normals:
        normals_arr = np.array(normals, dtype=np.float32)
    else:
        normals_arr = np.zeros((len(vertices_arr), 3), dtype=np.float32)

    if uvs:
        uvs_arr = np.array(uvs, dtype=np.float32)
    else:
        uvs_arr = np.zeros((len(vertices_arr), 2), dtype=np.float32)

    return Mesh(
        vertices=vertices_arr,
        indices=indices_arr,
        normals=normals_arr,
        uvs=uvs_arr,
    )



