import os
import time
import threading

import numpy as np
import moderngl


class ShaderProgram:
    def __init__(self, ctx: moderngl.Context, vert_path: str, frag_path: str):
        self.ctx = ctx
        self.vert_path = vert_path
        self.frag_path = frag_path
        self.program: moderngl.Program | None = None
        self._last_vert_mtime: float = 0.0
        self._last_frag_mtime: float = 0.0
        self._load()

    def _read_source(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load(self) -> bool:
        try:
            vert_src = self._read_source(self.vert_path)
            frag_src = self._read_source(self.frag_path)
            self.program = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
            self._last_vert_mtime = os.path.getmtime(self.vert_path)
            self._last_frag_mtime = os.path.getmtime(self.frag_path)
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
            return True
        self.program = old
        return False

    def watch_and_reload(self, interval: float = 0.5) -> None:
        def _watcher():
            while True:
                try:
                    vert_mtime = os.path.getmtime(self.vert_path)
                    frag_mtime = os.path.getmtime(self.frag_path)
                    if vert_mtime != self._last_vert_mtime or frag_mtime != self._last_frag_mtime:
                        self.reload()
                except Exception:
                    pass
                time.sleep(interval)

        t = threading.Thread(target=_watcher, daemon=True)
        t.start()

    def use(self) -> None:
        pass

    def set_uniform(self, name: str, value) -> None:
        if self.program is not None and name in self.program:
            if isinstance(value, np.ndarray):
                self.program[name] = value.ravel().tolist()
            else:
                self.program[name] = value
