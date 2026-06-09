from PIL import Image
import numpy as np
import moderngl


class Texture:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.texture: moderngl.Texture | None = None

    def _release(self) -> None:
        if self.texture is not None:
            self.texture.release()
            self.texture = None

    def load_file(self, path: str) -> None:
        self._release()
        img = Image.open(path).convert('RGBA')
        data = np.array(img, dtype=np.uint8)
        self.texture = self.ctx.texture(img.size, 4, data.tobytes())
        self.texture.build_mipmaps()

    def load_numpy(self, data: np.ndarray) -> None:
        self._release()
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got shape {data.shape}")
        h, w, c = data.shape
        if c == 3:
            rgba = np.ones((h, w, 4), dtype=np.uint8) * 255
            rgba[:, :, :3] = data[:, :, :3]
            data = rgba
        elif c == 1:
            rgba = np.ones((h, w, 4), dtype=np.uint8) * 255
            rgba[:, :, :3] = data[:, :, 0]
            data = rgba
        elif c != 4:
            raise ValueError(f"Unsupported channels: {c}, expected 3 or 4")
        self.texture = self.ctx.texture((w, h), 4, data.tobytes())
        self.texture.build_mipmaps()

    def bind(self, slot: int = 0) -> None:
        if self.texture is not None:
            self.texture.use(slot)

    def release(self) -> None:
        self._release()

    @property
    def gl_handle(self) -> int:
        if self.texture is not None:
            return self.texture.glo
        return 0
