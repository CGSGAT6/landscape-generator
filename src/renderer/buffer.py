import moderngl
import numpy as np

class Buffer:
    buffer: moderngl.Buffer | None = None
    size: int = 0

    def __init__(self, rnd, data: np.ndarray | None = None, reserve=0):
        self.rnd = rnd
        self.buffer = self.rnd.ctx.buffer(data=data, reserve=reserve, dynamic=True)
        if data is None:
            self.size = reserve
        else:
            self.size = data.nbytes

    @property    
    def is_valid(self):
        return not self.buffer is None

    def read(self, size: int = -1, *, offset: int = 0):
        if not self.is_valid:
            return None
        
        if size == -1:
            size = self.size

        if offset > self.size:
            return None

        if offset + size > self.size:
            size -= self.size - offset
        
        return self.buffer.read(size=size, offset=offset)
    
    def read_into(self, buffer, size: int = -1, offset: int = 0, write_offset: int = 0):
        if not self.is_valid:
            return
        if size == -1:
            size = self.size

        if offset > self.size:
            return

        if offset + size > self.size:
            size -= self.size - offset
        
        self.buffer.read_into(buffer=buffer, size=size, offset=offset, write_offset=write_offset)
    
    def resize(self, new_size):
        if not self.buffer is None:
            self.buffer.release()

        self.size = new_size
        self.buffer = self.rnd.ctx.buffer(reserve=self.size, dynamic=True)

    def write(self, data: np.ndarray, offset: int = 0):
        if offset + data.nbytes > self.size:
            self.resize(new_size=offset + data.nbytes)
        self.buffer.write(data=data, offset=offset)
        