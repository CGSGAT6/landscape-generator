from pathlib import Path

import moderngl
import numpy as np
import pyrr

from .primitive import Primitive


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


class Model:
    def __init__(self, rnd, prims=None, file_path=None):
        self.rnd = rnd
        self.prims = []
        self.bbox_min = None
        self.bbox_max = None
        if prims is not None:
            for p in prims:
                self.prims.append((p, pyrr.Matrix44.identity()))
            self.compute_bbox()
        elif file_path is not None:
            self.add_from_file(file_path)

    def add_prims(self, prims, local_matrix=pyrr.Matrix44.identity()):
        for p in prims:
            self.prims.append((p, local_matrix))
        self.compute_bbox()
      
    def clear_prims(self):
        self.prims.clear()
        self.compute_bbox()
        
    def __getitem__(self, idx: int):
        return self.prims[idx][0]

    def set_matrix(self, idx: int, local_matrix=pyrr.Matrix44.identity()):
        if idx < len(self.prims):
            self.prims[idx] = (self.prims[idx][0], local_matrix)

    # ------------------------------------------------------------------ OBJ
    def _parse_obj_face(self, parts):
        verts = []
        for p in parts[1:]:
            indices = p.split('/')
            vi = int(indices[0]) - 1
            vti = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else -1
            vni = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else -1
            verts.append((vi, vti, vni))
        return verts

    @staticmethod
    def _triangulate_face(face_verts):
        if len(face_verts) < 3:
            return []
        tris = []
        for i in range(1, len(face_verts) - 1):
            tris.append([face_verts[0], face_verts[i], face_verts[i + 1]])
        return tris

    # --------------------------------------------------------------- build
    def _build_smooth(self, positions, normals, uvs, face_verts_list):
        ref_order = []
        ref_map = {}

        for face_verts in face_verts_list:
            tris = self._triangulate_face(face_verts)
            for tri in tris:
                for ref in tri:
                    if ref not in ref_map:
                        ref_map[ref] = len(ref_order)
                        ref_order.append(ref)

        verts = []
        has_vn = len(normals) > 0
        for vi, vti, vni in ref_order:
            pos = positions[vi]
            nrm = normals[vni] if has_vn and 0 <= vni < len(normals) else [0.0, 0.0, 0.0]
            uv = uvs[vti] if 0 <= vti < len(uvs) else [0.0, 0.0]
            verts.extend([*pos, *nrm, *uv])

        vertices = np.array(verts, dtype=np.float32)

        indices = []
        for face_verts in face_verts_list:
            tris = self._triangulate_face(face_verts)
            for tri in tris:
                for ref in tri:
                    indices.append(ref_map[ref])
        indices_arr = np.array(indices, dtype=np.uint32)

        if not has_vn:
            norms_computed = compute_normals(
                vertices.reshape(-1, 8)[:, :3],
                indices_arr.reshape(-1, 3),
            )
            for i in range(len(ref_order)):
                vertices[i * 8 + 3:i * 8 + 6] = norms_computed[i]

        return vertices, indices_arr

    def _build_flat(self, positions, uvs, face_verts_list):
        verts = []
        for face_verts in face_verts_list:
            tris = self._triangulate_face(face_verts)
            for tri in tris:
                p0 = positions[tri[0][0]]
                p1 = positions[tri[1][0]]
                p2 = positions[tri[2][0]]

                fn = np.cross(p1 - p0, p2 - p0)
                fn_len = np.linalg.norm(fn)
                fn = fn / fn_len if fn_len > 0 else np.array([0.0, 0.0, 1.0])

                for vi, vti, vni in tri:
                    pos = positions[vi]
                    uv = uvs[vti] if 0 <= vti < len(uvs) else [0.0, 0.0]
                    verts.extend([*pos, *fn, *uv])

        return np.array(verts, dtype=np.float32)

    # --------------------------------------------------------- parse .mtl
    def _parse_mtl(self, mtl_path):
        mtl_dir = mtl_path.parent
        materials = {}
        current = None

        for line in mtl_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'newmtl':
                current = parts[1]
                materials[current] = {
                    'Ka': [0.47, 0.30, 0.18],
                    'Kd': [0.47, 0.30, 0.18],
                    'Ks': [0.47, 0.30, 0.18],
                    'Ns': 500.0,
                    'd': 0.0,
                    'map_Kd': None,
                    'map_Ka': None,
                    'map_Ks': None,
                }
            elif current is not None:
                if parts[0] == 'Ka':
                    materials[current]['Ka'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif parts[0] == 'Kd':
                    materials[current]['Kd'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif parts[0] == 'Ks':
                    materials[current]['Ks'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif parts[0] == 'Ns':
                    materials[current]['Ns'] = float(parts[1])
                elif parts[0] == 'd':
                    materials[current]['d'] = 1.0 - float(parts[1])
                elif parts[0] == 'Tr':
                    materials[current]['d'] = float(parts[1])
                elif parts[0] == 'map_Kd' and len(parts) > 1 and parts[1].strip():
                    materials[current]['map_Kd'] = str(mtl_dir / parts[1])
                elif parts[0] == 'map_Ka' and len(parts) > 1 and parts[1].strip():
                    materials[current]['map_Ka'] = str(mtl_dir / parts[1])
                elif parts[0] == 'map_Ks' and len(parts) > 1 and parts[1].strip():
                    materials[current]['map_Ks'] = str(mtl_dir / parts[1])

        return materials

    # -------------------------------------------------------- add_from_file
    def add_from_file(self, fname, flat=False):
        obj_path = Path(fname)
        base_dir = obj_path.parent

        # ---- parse OBJ ----
        positions = []
        normals = []
        uvs = []
        groups = []
        current_faces = []
        current_mtl = None
        mtllib_path = None

        for line in obj_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn':
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt':
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f':
                current_faces.append(self._parse_obj_face(parts))
            elif parts[0] == 'usemtl':
                if current_faces:
                    groups.append((current_mtl, current_faces))
                    current_faces = []
                current_mtl = parts[1]
            elif parts[0] == 'mtllib':
                mtllib_path = parts[1]

        if current_faces:
            groups.append((current_mtl, current_faces))

        positions = np.array(positions, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32) if normals else np.empty((0, 3), dtype=np.float32)
        uvs = np.array(uvs, dtype=np.float32) if uvs else np.empty((0, 2), dtype=np.float32)

        if len(positions) > 0:
            self.bbox_min = positions.min(axis=0)
            self.bbox_max = positions.max(axis=0)

        # ---- parse MTL ----
        mtl_data = {}
        if mtllib_path is not None:
            mtl_full = base_dir / mtllib_path
            if mtl_full.is_file():
                mtl_data = self._parse_mtl(mtl_full)

        # ---- build primitives per group ----
        for mtl_name, face_verts_list in groups:
            mtl_obj = None
            if mtl_name is not None and mtl_name in mtl_data:
                info = mtl_data[mtl_name]
                tex_list = []
                if info['map_Kd'] is not None:
                    tex_list.append(self.rnd.create_texture(img_path=info['map_Kd']))

                mtl_obj = self.rnd.get_material(
                    mtl_name,
                    ka=pyrr.Vector3(info['Ka']),
                    kd=pyrr.Vector3(info['Kd']),
                    ks=pyrr.Vector3(info['Ks']),
                    ph=info['Ns'] / 1000.0,
                    tr=info['d'],
                    tex=tex_list,
                )
            else:
                mtl_obj = self.rnd.get_material("default")

            if flat:
                verts = self._build_flat(positions, uvs, face_verts_list)
                prim = self.rnd.create_prim(vertices=verts, mtl=mtl_obj)
            else:
                verts, idx = self._build_smooth(positions, normals, uvs, face_verts_list)
                prim = self.rnd.create_prim(vertices=verts, indeces=idx, mtl=mtl_obj)

            self.prims.append((prim, pyrr.Matrix44.identity()))

    # ---------------------------------------------------------------- bbox
    def compute_bbox(self):
        if not self.prims:
            self.bbox_min = None
            self.bbox_max = None
            return
        bmin = np.full(3, np.inf, dtype=np.float32)
        bmax = np.full(3, -np.inf, dtype=np.float32)
        for p, _ in self.prims:
            bmin = np.minimum(bmin, p.bbox_min)
            bmax = np.maximum(bmax, p.bbox_max)
        self.bbox_min = bmin
        self.bbox_max = bmax

    # --------------------------------------------------------------- render
    def render(self, world: pyrr.Matrix44 | None = None):
        if world is None:
            world = pyrr.Matrix44.identity()

        for p, l in self.prims:
            self.rnd.render_prim(p, l * world)
