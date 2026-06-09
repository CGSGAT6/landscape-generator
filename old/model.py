import numpy as np
import moderngl

from .shader import ShaderProgram


class Model:
    def __init__(self, ctx: moderngl.Context, primitive_mode: int = moderngl.TRIANGLES):
        self.ctx = ctx
        self.vbo: moderngl.Buffer | None = None
        self.ibo: moderngl.Buffer | None = None
        self._vao_cache: dict[int, moderngl.VertexArray] = {}
        self.vertex_count: int = 0
        self._normals_uv_size: int = 0
        self._primitive_mode = primitive_mode

    def _release_buffers(self) -> None:
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        if self.ibo is not None:
            self.ibo.release()
            self.ibo = None
        for vao in self._vao_cache.values():
            vao.release()
        self._vao_cache.clear()
        self.vertex_count = 0

    def load_from_mesh(self, vertices: np.ndarray, indices: np.ndarray,
                       normals: np.ndarray, uvs: np.ndarray) -> None:
        self._release_buffers()

        n = len(vertices)
        vertex_data = np.zeros((n, 8), dtype=np.float32)
        vertex_data[:, :3] = vertices
        vertex_data[:, 3:6] = normals
        vertex_data[:, 6:8] = uvs

        self.vbo = self.ctx.buffer(vertex_data.tobytes())
        self.ibo = self.ctx.buffer(indices.astype(np.uint32).tobytes())
        self.vertex_count = indices.size

    def load_obj(self, path: str) -> None:
        vertices: list[list[float]] = []
        normals: list[list[float]] = []
        uvs: list[list[float]] = []
        faces_verts: list[list[int]] = []
        faces_uvs: list[list[int]] = []
        faces_normals: list[list[int]] = []

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vt':
                    uvs.append([float(parts[1]), float(parts[2])])
                elif parts[0] == 'f':
                    v_face: list[int] = []
                    vt_face: list[int] = []
                    vn_face: list[int] = []
                    for token in parts[1:]:
                        indices_str = token.split('/')
                        v_face.append(int(indices_str[0]) - 1)
                        if len(indices_str) > 1 and indices_str[1]:
                            vt_face.append(int(indices_str[1]) - 1)
                        if len(indices_str) > 2 and indices_str[2]:
                            vn_face.append(int(indices_str[2]) - 1)
                    faces_verts.append(v_face)
                    faces_uvs.append(vt_face)
                    faces_normals.append(vn_face)

        vert_arr = np.array(vertices, dtype=np.float32)
        norm_arr = np.array(normals, dtype=np.float32) if normals else np.zeros_like(vert_arr)
        uv_arr = np.array(uvs, dtype=np.float32) if uvs else np.zeros((len(vertices), 2), dtype=np.float32)

        indices_list: list[int] = []
        for fv, ft, fn in zip(faces_verts, faces_uvs, faces_normals):
            for i in range(1, len(fv) - 1):
                indices_list.extend([fv[0], fv[i], fv[i + 1]])

        idx_arr = np.array(indices_list, dtype=np.uint32).reshape(-1, 3)

        self.load_from_mesh(vert_arr, idx_arr, norm_arr, uv_arr)

    def update_vertices(self, vertices: np.ndarray) -> None:
        if self.vbo is None:
            return
        current = np.frombuffer(self.vbo.read(), dtype=np.float32).reshape(-1, 8)
        n = len(vertices)
        vertex_data = np.zeros((n, 8), dtype=np.float32)
        vertex_data[:, :3] = vertices
        if n == len(current):
            vertex_data[:, 3:] = current[:, 3:]
        self.vbo.write(vertex_data.tobytes())

    @property
    def primitive_mode(self) -> int:
        return self._primitive_mode

    @primitive_mode.setter
    def primitive_mode(self, mode: int) -> None:
        self._primitive_mode = mode

    def draw(self, shader: ShaderProgram) -> None:
        if self.vbo is None or self.ibo is None or shader.program is None:
            return

        prog_id = id(shader.program)
        if prog_id not in self._vao_cache:
            vao = self.ctx.vertex_array(
                shader.program,
                [(self.vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
                self.ibo
            )
            self._vao_cache[prog_id] = vao

        self._vao_cache[prog_id].render(self._primitive_mode)

    def release(self) -> None:
        self._release_buffers()
