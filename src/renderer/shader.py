import numpy as np
import moderngl
class Shader:
    def __init__(self, rnd, vert_path: str, frag_path: str):
        self.rnd = rnd
        self.vert_path = vert_path
        self.frag_path = frag_path
        self.program: moderngl.Program | None = None
        self.update_frame = -1
        self._is_valid = self._load()
        if self._is_valid:
            self.update_frame = self.rnd.cur_frame  

    @property
    def is_valid(self):
        return self._is_valid
    def _read_source(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load(self) -> bool:
        try:
            vert_src = self._read_source(self.vert_path)
            frag_src = self._read_source(self.frag_path)
            self.program = self.rnd.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
            return True
        except moderngl.Error as e:
            print(f"[ShaderProgram] Compile error: {e}")
            return False
        except Exception as e:
            print(f"[ShaderProgram] IO error: {e}")
            return False

    def reload(self) -> bool:
        old = self.program
        if self._load():
            self.update_frame = self.rnd.cur_frame
            return True
        self.program = old
        return False

    def _get_attribute_by_location(self, target_location):
        for name in self.program:
            member = self.program[name]
            if isinstance(member, moderngl.Attribute) and member.location == target_location:
                return member
        return None

    def set_uniform(self, name: str, value) -> None:
        if self.program is not None and name in self.program:
            if isinstance(value, np.ndarray):
                self.program[name] = value.ravel().tolist()
            else:
                self.program[name] = value
