from .constants import VERTEX_SIZE
import moderngl
import numpy as np
class Primitive:
    vbuf: moderngl.Buffer | None = None
    ibuf: moderngl.Buffer | None = None
    vao: moderngl.VertexArray | None = None
    noofv: int = 0
    noofi: int = 0


    def __init__(self, rnd, vertices: np.ndarray,
                 indeces: np.ndarray | None  = None, shd = None, mtl = None):
        self.rnd = rnd
        if shd is None:
            shd = rnd.get_shader("default")
        if mtl is None:
            mtl = rnd.get_material("default")
        self.shd = shd
        self.mtl = mtl
        self.noofv = int(vertices.nbytes / VERTEX_SIZE)
        if indeces is not None:
            self.noofi = int(indeces.nbytes / 4)
        self._compute_bbox(vertices)
        self._update_buffers(vertices=vertices, indeces=indeces)
        self._update_vao()

    @property
    def is_valid(self):
        return not self.shd is None and self.shd.is_valid and not self.vao is None and not self.mtl is None
    
    def _update_vao(self):
        if not self.shd.is_valid or self.vbuf is None or (self.noofi > 0 and self.ibuf is None):
            return

        vertex_format = (
            ("3f", 0, "3x4"),
            ("3f", 1, "3x4"),
            ("2f", 2, "2x4"),
        )

        fstr = ""
        locs = ()

        for sign, loc, padd in vertex_format:
            if not (self.shd._get_attribute_by_location(loc) is None):
                fstr += sign + " "
                locs += (loc,)
            else:
                fstr += padd + " "
        if self.noofi > 0:
            self.vao = self.rnd.ctx.vertex_array(program = self.shd.program,
                                             content=
                                             [
                                                 #(self.vbuf, "3f 3f 2f", 0, 1, 2)
                                                 (self.vbuf, fstr, *locs)
                                             ],
                                             index_buffer=self.ibuf,
                                             index_element_size=4)
        else:
            self.vao = self.rnd.ctx.vertex_array(program = self.shd.program,
                                             content=
                                             [
                                                 #(self.vbuf, "3f 3f 2f", 0, 1, 2)
                                                 (self.vbuf, fstr, *locs)
                                             ])    


    def _update_buffers(self, vertices: np.ndarray, indeces:np.ndarray | None  = None):
        self.vbuf = self.rnd.ctx.buffer(reserve=vertices.nbytes)
        self.vbuf.write(vertices.tobytes())
        if indeces is not None:
            self.ibuf = self.rnd.ctx.buffer(reserve=indeces.nbytes)
            self.ibuf.write(indeces.tobytes())
    def _compute_bbox(self, vertices: np.ndarray):
        pos = vertices.reshape(-1, 8)[:, :3]
        self.bbox_min = pos.min(axis=0)
        self.bbox_max = pos.max(axis=0)

    def _render(self):
        if not self.is_valid:
            return
        self.mtl.use()
        if (self.noofi > 0):
            self.vao.render(mode=moderngl.TRIANGLES, vertices=self.noofi)
        else:
            self.vao.render(mode=moderngl.TRIANGLES, vertices=self.noofv)
