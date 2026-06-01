import pyrr
import numpy as np
from .constants import MATERIAL_BINDING
class Material:
    MATETRIAL_BUF_SIZE = 4*4*4

    def __init__(self, rnd, name: str,
                 ka: pyrr.Vector3 = pyrr.Vector3([0.47, 0.30, 0.18]),
                 kd: pyrr.Vector3 = pyrr.Vector3([0.47, 0.30, 0.18]),
                 ks: pyrr.Vector3 = pyrr.Vector3([0.47, 0.30, 0.18]),
                 ph: float = 0.5, tr: float = 0):
        self.rnd = rnd
        self.ka = ka
        self.ks = ks
        self.kd = kd
        self.ph = ph     
        self.tr = tr

        self.buf = rnd.ctx.buffer(reserve=self.MATETRIAL_BUF_SIZE, dynamic=True)
        self.update_buffers()
    def update_buffers(self):
        vectors = np.array([self.ka, self.kd, self.ks,
                   pyrr.Vector3([0, 0, 0])], dtype=np.float32)
        c4 = np.array([self.ph, self.tr, 0, 0], dtype=np.float32)
        buf = np.column_stack((vectors, c4))

        self.buf.write(buf.tobytes())
    
    def update(self, ka = None,
                 kd = None,
                 ks = None,
                 ph = None, tr = None):
        if (ka != None):
            self.ka = ka
        if (ks != None):
            self.ks = ks
        if (kd != None):
            self.kd = kd
        if (ph != None):
            self.ph = ph     
        if (tr != None):
            self.tr = tr
        self.update_buffers()

    def use(self):
        self.buf.bind_to_uniform_block(binding=MATERIAL_BINDING)