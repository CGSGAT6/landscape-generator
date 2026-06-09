import numpy as np
import moderngl

from .texture import Texture


class Framebuffer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.fbo: moderngl.Framebuffer | None = None
        self._color_tex: moderngl.Texture | None = None
        self._depth_tex: moderngl.Texture | None = None
        self._create()

    def _release(self) -> None:
        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None
        if self._color_tex is not None:
            self._color_tex.release()
            self._color_tex = None
        if self._depth_tex is not None:
            self._depth_tex.release()
            self._depth_tex = None

    def _create(self) -> None:
        self._color_tex = self.ctx.texture((self.width, self.height), 4, dtype='u1')
        self._depth_tex = self.ctx.depth_texture((self.width, self.height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self._color_tex],
            depth_attachment=self._depth_tex
        )

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0) -> None:
        if self._color_tex is not None:
            bg = np.full(
                (self.height, self.width, 4),
                [int(r * 255), int(g * 255), int(b * 255), int(a * 255)],
                dtype=np.uint8,
            )
            self._color_tex.write(bg.tobytes())

    def bind(self) -> None:
        if self.fbo is not None:
            self.fbo.use()

    def unbind(self) -> None:
        self.ctx.screen.use()

    def resize(self, width: int, height: int) -> None:
        self._release()
        self.width = width
        self.height = height
        self._create()

    def get_color_texture(self) -> Texture:
        tex = Texture(self.ctx)
        tex.texture = self._color_tex
        return tex

    def read(self, components: int = 4, dtype: str = 'u1') -> bytes:
        if self.fbo is not None:
            return self.fbo.read(components=components, dtype=dtype)
        return b''

    def release(self) -> None:
        self._release()
