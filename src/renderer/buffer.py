import moderngl
import numpy as np

class Buffer:
    buffer: moderngl.Buffer | None = None
    size: int = 0

    def __init__(self, rnd, data: bytes | np.ndarray | None = None, reserve=0):
        self.rnd = rnd
        if data is None:
            self.size = reserve
            self.buffer = self.rnd.ctx.buffer(reserve=reserve, dynamic=True)
        elif isinstance(data, bytes):
            self.size = len(data)
            self.buffer = self.rnd.ctx.buffer(data=data, dynamic=True)
        elif isinstance(data, np.ndarray):
            self.size = data.nbytes
            self.buffer = self.rnd.ctx.buffer(data=data.tobytes(), dynamic=True)

    @property    
    def is_valid(self):
        return not self.buffer is None

    def read(self, size: int = -1, offset: int = 0):
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

    def write(self, data: np.ndarray | bytes, offset: int = 0):
        end = 0
        
        if isinstance(data, bytes):
            end = offset + len(data)
        elif isinstance(data, np.ndarray):
            end = offset + data.nbytes
        
        if end > self.size:
            self.resize(new_size=end)
        
        if isinstance(data, bytes):
            self.buffer.write(data=data, offset=offset)
        elif isinstance(data, np.ndarray):
            self.buffer.write(data=data.tobytes(), offset=offset)

    def bind(self, binding: int = 0, offset: int = 0, size: int = -1):
        self.buffer.bind_to_storage_buffer(binding=binding, offset=offset, size=size)
        