from .constants import *
from filepath import ASSETS_DIR, SHADERS_DIR
from .shader import Shader
from .material import Material
from .primitive import Primitive
from .camera import Camera
from .buffer import Buffer
from .texture import Texture
from .model import Model
from .utils import *

import numpy as np
import moderngl
import pyrr

class RenderEngine:
    clear_color = (0.18, 0.30, 0.47, 1.0)

    def __init__(self):
        self.ctx: moderngl.Context | None = None
        self.shaders_dict = {}
        self.materials_dict = {}
        self.primitives_dict = {}
        self.buffers_dict = {}
        self.textures_dict = {}
        self.models_dict = {}
        self.cur_frame = 0
        self.width = 470
        self.height = 470
        
    PRIM_BUF_SIZE = 16 * 4 * 3
    CAMERA_BUF_SIZE = 16 * (4 * 3 + 2)
    SYNC_BUF_SIZE = 4 * 4     

    def init(self):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prim_buf = self.create_buffer(reserve=self.PRIM_BUF_SIZE)
        self.sync_buf = self.create_buffer(reserve=self.SYNC_BUF_SIZE)
        self.camera_buf = self.create_buffer(reserve=self.CAMERA_BUF_SIZE)
        self.camera = Camera()

        self.sync_buf.bind(SYNC_BINDING)
        self.camera_buf.bind(CAMERA_BINDING)
        self.prim_buf.bind(PRIM_BINDING)
    def create_standalone(self, width: int, height: int) -> None:
        self.ctx = moderngl.create_standalone_context()
        self.is_standalone = True
        self.resize(width, height)

    def create_from_window(self, ctx: moderngl.Context | None = None) -> None:
        if ctx is not None:
            self.ctx = ctx
        else:
            self.ctx = moderngl.create_context(require=430)
        ctx.gc_mode = "context_gc"
        self.is_standalone = False

    @property
    def is_valid(self) -> bool:
        return self.ctx is not None

    def set_clear_color(self, color):
        self.clear_color = color

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
        if self.is_standalone:
            self.ctx.release()
    
    def _update_camera_buf(self):
        
        pos_padded = np.pad(self.camera._get_position(), (0, 1), mode='constant')  # [1,2,3,0]
        forw_padded = np.pad(self.camera._get_forward(), (0, 1), mode='constant')
        combined = np.concatenate([self.camera.get_vp_matrix().ravel(),
                                   self.camera.get_view_matrix().ravel(),
                                   self.camera.get_projection_matrix().ravel(), 
                                   pos_padded, forw_padded], dtype=np.float32)

        self.camera_buf.write(combined.tobytes())
    def _update_sync_buf(self):
        sync_arr = np.array([float(self.time), float(self.delta_time), float(self.width), float(self.height)], dtype=np.float32)
        
        self.sync_buf.write(sync_arr.tobytes())
        
    def begin_frame(self, time, delta_time):
        self.screen_fbo.use()
        self.clear(self.clear_color)
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
                     ph: float = 0.5, tr: float = 0, tex=[]):
        mtl = self.materials_dict.get(mtl_name)
        if mtl is None:
            mtl = Material(self, name=mtl_name, ka=ka, kd=kd, ks=ks, ph=ph, tr=tr, tex=tex)
            self.materials_dict[mtl_name] = mtl
        return mtl    

    def render_prim(self, prim: Primitive, world: pyrr.Matrix44 | None = None):
        if prim is None or not prim.is_valid:
            return
    
        if world is None:
            world = pyrr.Matrix44.identity()
        winv = ~(world.T)

        arr = self.camera_buf.read(size=64, offset=64*2)

        db = np.frombuffer(buffer=arr, dtype=np.float32).reshape(4,4)

        prim_arr = np.array([world, winv, self.camera.get_vp_matrix() * world], dtype=np.float32)
        self.prim_buf.write(prim_arr.tobytes())

        prim._render()

    def create_prim(self, vertices: np.ndarray,
                    indeces: np.ndarray | None  = None, shd = None, mtl = None, name = None):
        pr = Primitive(rnd=self, vertices=vertices, indeces=indeces, shd=shd, mtl=mtl)

        if name is not None:
            self.primitives_dict[name] = pr
        return pr
    
    def get_primitive(self, name):
        return self.primitives_dict.get(name)

    def create_buffer(self, name = None, data: np.ndarray | None = None, reserve=0):
        buf = Buffer(self, data=data, reserve=reserve)

        if name is not None:
            self.buffers_dict[name] = buf

        return buf
    
    def get_buffer(self, name):
        return self.buffers_dict.get(name)

    def create_texture(self, name = None, img_path = None, data = None):
        tex = Texture(self, img_path=img_path, data=data)

        if name is not None:
            self.textures_dict[name] = tex
        
        return tex
    def get_texture(self, name):
        return self.textures_dict.get(name)

    def create_model(self, name=None, prims=None, file_path=None):
        mdl = Model(self, prims=prims, file_path=file_path)

        if name is not None:
            self.models_dict[name] = mdl
        
        return mdl
    
    def get_model(self, name):
        return self.models_dict.get(name)

    def reload_shaders(self):
        for n in self.shaders_dict:
            self.shaders_dict[n].reload()

    def camera_orbit(self, delta_yaw: float, delta_pitch: float) -> None:
        self.camera.orbit(delta_yaw, delta_pitch)
        self._update_camera_buf()

    def camera_zoom(self, delta: float) -> None:
        self.camera.zoom(delta)
        self._update_camera_buf()

    def camera_pan(self, delta_x: float, delta_y: float) -> None:
        self.camera.pan(delta_x, delta_y)
        self._update_camera_buf()    

#        self.shaders_dict = {}
#        self.materials_dict = {}
#        self.primitives_dict = {}
#        self.buffers_dict = {}
#        self.textures_dict = {}
#        self.models_dict = {}


    def clear_shaders_cash(self):
        self.shaders_dict.clear()

    def clear_materials_cash(self):
        self.materials_dict.clear()

    def clear_primitives_cash(self):
        self.primitives_dict.clear()

    def clear_buffers_cash(self):
        self.buffers_dict.clear()

    def clear_textures_cash(self):
        self.textures_dict.clear()

    def clear_models_cash(self):
        self.models_dict.clear()

    def clear_resource_cash(self):
        self.clear_shaders_cash()
        self.clear_materials_cash()
        self.clear_primitives_cash()
        self.clear_buffers_cash()
        self.clear_textures_cash()
        self.clear_models_cash()