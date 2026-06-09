import pyrr
import numpy as np
from .constants import MATERIAL_BINDING
from .buffer import Buffer
from .texture import Texture
class Material:
    MATETRIAL_BUF_SIZE = 6*4*4
    MAX_TEX = 8

    def __init__(self, rnd, name: str,
                 ka: pyrr.Vector3 = pyrr.Vector3([0.47, 0.30, 0.18]),
                 kd: pyrr.Vector3 = pyrr.Vector3([0.47, 0.30, 0.18]),
                 ks: pyrr.Vector3 = pyrr.Vector3([0.47, 0.30, 0.18]),
                 ph: float = 0.5, tr: float = 0, tex=[]):
        self.rnd = rnd
        self.ka = ka
        self.ks = ks
        self.kd = kd
        self.ph = ph     
        self.tr = tr

        self.tex = [None] * self.MAX_TEX

        for i in range(min(len(tex), self.MAX_TEX)):
            if isinstance(tex[i], Texture):
                self.tex[i] = tex[i]
            else:
                self.tex[i] = self.rnd.create_texture(name=tex[i], img_path=tex[i])

        #self.buf = rnd.ctx.buffer(reserve=self.MATETRIAL_BUF_SIZE, dynamic=True)
        self.buf = rnd.create_buffer(reserve=self.MATETRIAL_BUF_SIZE)
        self.update_buffers()
        
    def set_texture(self, texture: Texture, idx: int) -> None:
        if not isinstance(texture, Texture):
            raise TypeError("Expected a Texture instance")
        if idx < 0 or idx >= self.MAX_TEX:
            raise IndexError(f"idx must be 0..{self.MAX_TEX - 1}")
        self.tex[idx] = texture
        self.update_buffers()

    def update_buffers(self):
        vectors = np.array([self.ka, self.kd, self.ks,
                   pyrr.Vector3([0, 0, 0])], dtype=np.float32)
        c4 = np.array([self.ph, self.tr, 0, 0], dtype=np.float32)
        buf = np.column_stack((vectors, c4))

        self.buf.write(buf.tobytes(), offset=0)

        tex_mask = np.zeros(8, dtype=np.int32)
        for i in range(self.MAX_TEX):
            if self.tex[i] is not None:
                tex_mask[i] = 1
        self.buf.write(tex_mask.tobytes(), offset=64)
    
    def update(self, ka = None,
                 kd = None,
                 ks = None,
                 ph = None, tr = None):
        if (ka is not None):
            self.ka = ka
        if (ks is not None):
            self.ks = ks
        if (kd is not None):
            self.kd = kd
        if (ph is not None):
            self.ph = ph     
        if (tr is not None):
            self.tr = tr
        self.update_buffers()

    def use(self):
        self.buf.bind(binding=MATERIAL_BINDING)
        for i in range(self.MAX_TEX):
            if self.tex[i] is not None:
                self.tex[i].bind(i)