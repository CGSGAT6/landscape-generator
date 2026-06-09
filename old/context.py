import moderngl


class RendererContext:
    def __init__(self):
        self.ctx: moderngl.Context | None = None

    def create_standalone(self, width: int, height: int) -> None:
        self.ctx = moderngl.create_standalone_context()
        self.resize(width, height)

    def create_from_window(self, ctx: moderngl.Context | None = None) -> None:
        if ctx is not None:
            self.ctx = ctx
        else:
            self.ctx = moderngl.create_context()

    @property
    def is_valid(self) -> bool:
        return self.ctx is not None

    def resize(self, width: int, height: int) -> None:
        if self.ctx is not None:
            self.ctx.viewport = (0, 0, width, height)

    def clear(self, color: tuple | None = None) -> None:
        if self.ctx is not None:
            if color is not None:
                self.ctx.clear(*color)
            else:
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
