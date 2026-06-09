from dataclasses import dataclass


@dataclass
class RenderSettings:
    background_color: tuple = (0.1, 0.1, 0.15, 1.0)
    wireframe: bool = False
    enable_fog: bool = True
    fog_density: float = 0.02
