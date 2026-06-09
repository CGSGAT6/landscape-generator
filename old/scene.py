from typing import Optional

import numpy as np

from .camera import Camera
from .shader import ShaderProgram
from .model import Model
from .settings import RenderSettings


class Scene:
    def __init__(self, ctx):
        self.ctx = ctx
        self._models: dict[str, Model] = {}
        self._transforms: dict[str, np.ndarray] = {}
        self._light_direction = np.array([0.5, -1.0, -0.3], dtype=np.float32)
        self._light_color = (1.0, 1.0, 1.0)
        self._light_intensity = 1.0

    def add_model(self, name: str, model: Model) -> None:
        self._models[name] = model
        self._transforms[name] = np.eye(4, dtype=np.float32)

    def remove_model(self, name: str) -> None:
        model = self._models.pop(name, None)
        if model is not None:
            model.release()
        self._transforms.pop(name, None)

    def get_model(self, name: str) -> Optional[Model]:
        return self._models.get(name)

    def set_model_transform(self, name: str, matrix: np.ndarray) -> None:
        if name in self._transforms:
            self._transforms[name] = matrix

    def set_directional_light(self, direction: tuple,
                              color: tuple = (1, 1, 1),
                              intensity: float = 1.0) -> None:
        self._light_direction = np.array(direction, dtype=np.float32)
        self._light_color = color
        self._light_intensity = intensity

    def render(self, camera: Camera,
               shader: ShaderProgram,
               settings: RenderSettings = RenderSettings(),
               extra_uniforms: dict | None = None) -> None:
        if settings.wireframe:
            self.ctx.wireframe = True

        shader.use()

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()

        shader.set_uniform('view', view)
        shader.set_uniform('projection', proj)

        light_dir = self._light_direction / np.linalg.norm(self._light_direction)
        shader.set_uniform('light_direction', tuple(light_dir))
        shader.set_uniform('light_color', self._light_color)
        shader.set_uniform('light_intensity', self._light_intensity)

        if extra_uniforms is not None:
            for name, value in extra_uniforms.items():
                shader.set_uniform(name, value)

        for name, model in self._models.items():
            model_matrix = self._transforms.get(name, np.eye(4, dtype=np.float32))
            shader.set_uniform('model', model_matrix)
            model.draw(shader)

        if settings.wireframe:
            self.ctx.wireframe = False
