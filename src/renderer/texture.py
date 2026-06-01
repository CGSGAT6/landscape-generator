import numpy as np
import PIL
from PIL import Image
import moderngl

class Texture:
    def __init__(self, rnd, img_path = None, data = None):
        self.rnd = rnd
        self.texture: moderngl.Texture | None = None
        if img_path is not None:
            self.load_file(img_path)
        else:
            self.load_numpy(data)
    
    def _release(self) -> None:
        if self.is_valid:
            self.texture.release()
            self.texture = None

    def load_file(self, path: str) -> None:
        self._release()
        try:
            img = Image.open(path).convert('RGBA')
        except FileNotFoundError:
            print("Файл не найден")
        except PIL.UnidentifiedImageError:
            print("Не удалось распознать формат изображения")
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
        else:
            data = np.array(img, dtype=np.uint8)
            self.texture = self.rnd.ctx.texture(img.size, 4, data.tobytes())
            self.texture.build_mipmaps()
            return
        self.texture = None

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
        self.texture = self.rnd.ctx.texture((w, h), 4, data.tobytes())
        self.texture.build_mipmaps()

    def bind(self, slot: int = 0) -> None:
        if self.is_valid:
            self.texture.use(slot)

    @property
    def is_valid(self):
        return self.texture is not None

    def release(self) -> None:
        self._release()
