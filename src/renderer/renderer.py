from .constants import *
from filepath import ASSETS_DIR, SHADERS_DIR
from .shader import Shader
from .material import Material
from .primitive import Primitive
from .camera import Camera
import numpy as np
import moderngl
import pyrr
class RenderEngine:
    def __init__(self):
        self.ctx: moderngl.Context | None = None
        self.shaders_dict = {}
        self.materials_dict = {}
        self.primitives_dict = {}
        self.cur_frame = 0
        self.width = 470
        self.height = 470
        
    PRIM_BUF_SIZE = 16 * 4 * 3
    CAMERA_BUF_SIZE = 16 * 4 * 3
    SYNC_BUF_SIZE = 4 * 4     

    def init(self):
        self.prim_buf = self.ctx.buffer(reserve=self.PRIM_BUF_SIZE, dynamic=True)
        self.sync_buf = self.ctx.buffer(reserve=self.SYNC_BUF_SIZE, dynamic=True)
        self.camera_buf = self.ctx.buffer(reserve=self.CAMERA_BUF_SIZE, dynamic=True)
        self.camera = Camera()

        self.sync_buf.bind_to_uniform_block(SYNC_BINDING)
        self.camera_buf.bind_to_uniform_block(CAMERA_BINDING)
        self.prim_buf.bind_to_uniform_block(PRIM_BINDING)
    def create_standalone(self, width: int, height: int) -> None:
        self.ctx = moderngl.create_standalone_context()
        self.resize(width, height)

    def create_from_window(self, ctx: moderngl.Context | None = None) -> None:
        if ctx is not None:
            self.ctx = ctx
        else:
            self.ctx = moderngl.create_context(require=430)
        ctx.gc_mode = "context_gc"

    @property
    def is_valid(self) -> bool:
        return self.ctx is not None

    def set_screen_fbo(self, screen_fbo):
        self.screen_fbo = screen_fbo

    def resize(self, width: int, height: int) -> None:
        if self.ctx is not None:
            self.ctx.viewport = (0, 0, width, height)
            self.camera.set_aspect(width, height)
            self.width = width
            self.height = height
            self._update_camera_buf()

    def clear(self, color: tuple | None = None) -> None:
        if self.ctx is not None:
            if color is not None:
                self.ctx.clear(*color)
            else:
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
    
    def close(self):
        self.ctx.gc()
    
    def _update_camera_buf(self):
        camera_arr = np.array([self.camera.get_view_matrix(),
                               self.camera.get_projection_matrix(),
                               self.camera.get_vp_matrix()], dtype=np.float32)
        self.camera_buf.write(camera_arr.tobytes())
    def _update_sync_buf(self):
        sync_arr = np.array([float(self.time), float(self.delta_time), float(self.width), float(self.height)], dtype=np.float32)
        
        self.sync_buf.write(sync_arr.tobytes())
        
    def begin_frame(self, time, delta_time):
        self.screen_fbo.use()
        self.clear((0.18, 0.30, 0.47, 1.0))
        self.time = time
        self.delta_time = delta_time
        self.cur_frame += 1
        self._update_sync_buf()
        
    def end_frame(self):
        pass
    """
    Resources
    """

    def get_shader(self, dir_name: str):
        shd = self.shaders_dict.get(dir_name)
        if shd is None:
            shd = Shader(self, SHADERS_DIR / dir_name / "vert.glsl", SHADERS_DIR / dir_name / "frag.glsl")
            self.shaders_dict[dir_name] = shd
        return shd    

    def get_material(self, mtl_name: str,
                     ka: pyrr.Vector3 = pyrr.Vector3([0.47, 0.30, 0.18]),
                     kd: pyrr.Vector3 = pyrr.Vector3([0.47, 0.30, 0.18]),
                     ks: pyrr.Vector3 = pyrr.Vector3([0.47, 0.30, 0.18]),
                     ph: float = 0.5, tr: float = 0):
        mtl = self.materials_dict.get(mtl_name)
        if mtl is None:
            mtl = Material(self, name=mtl_name, ka=ka, kd=kd, ks=ks, ph=ph, tr=tr)
            self.materials_dict[mtl_name] = mtl
        return mtl    

    def render_prim(self, prim: Primitive, world: pyrr.Matrix44 | None = None):
        if prim is None or not prim.is_valid:
            return
    
        if world is None:
            world = pyrr.Matrix44.identity()
        winv = (~world).T
        prim_arr = np.array([world, winv, world * self.camera.get_vp_matrix()], dtype=np.float32)
        self.prim_buf.write(prim_arr.tobytes())

        prim._render()
    def create_prim(self, vertices: np.ndarray,
                    indeces: np.ndarray | None  = None, shd = None, mtl = None):
        return Primitive(rnd=self, vertices=vertices, indeces=indeces, shd=shd, mtl=mtl)