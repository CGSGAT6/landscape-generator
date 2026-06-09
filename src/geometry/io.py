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
    face_refs = []

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
            refs = []
            for p in parts[1:]:
                indices = p.split("/")
                vi = int(indices[0]) - 1
                vti = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else -1
                vni = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else -1
                refs.append((vi, vti, vni))
            for i in range(1, len(refs) - 1):
                face_refs.append([refs[0], refs[i], refs[i + 1]])

    has_normals = len(normals) > 0
    has_uvs = len(uvs) > 0

    ref_map = {}
    ref_order = []
    for tri in face_refs:
        for ref in tri:
            if ref not in ref_map:
                ref_map[ref] = len(ref_order)
                ref_order.append(ref)

    verts = []
    norms = []
    uvs_exp = []
    for vi, vti, vni in ref_order:
        verts.append(positions[vi])
        if has_normals and 0 <= vni < len(normals):
            norms.append(normals[vni])
        else:
            norms.append([0.0, 0.0, 0.0])
        if has_uvs and 0 <= vti < len(uvs):
            uvs_exp.append(uvs[vti])
        else:
            uvs_exp.append([0.0, 0.0])

    indices = []
    for tri in face_refs:
        for ref in tri:
            indices.append(ref_map[ref])

    vertices_arr = np.array(verts, dtype=np.float32)
    indices_arr = np.array(indices, dtype=np.uint32).reshape(-1, 3)
    normals_arr = np.array(norms, dtype=np.float32)
    uvs_arr = np.array(uvs_exp, dtype=np.float32)

    if not has_normals:
        from .utils import compute_normals
        normals_arr = compute_normals(vertices_arr, indices_arr)

    return Mesh(
        vertices=vertices_arr,
        indices=indices_arr,
        normals=normals_arr,
        uvs=uvs_arr,
    )



