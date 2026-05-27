from PIL import Image
import numpy as np
import json
from .types import NoiseMap, PointSet


def save_noise_as_image(noise_map: NoiseMap, path: str,
                        colormap: str = "grayscale") -> None:
    data = (noise_map.data * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(data, mode='L')
    img.save(path)


def load_noise_from_image(path: str) -> NoiseMap:
    img = Image.open(path).convert('L')
    data = np.array(img, dtype=np.float32) / 255.0
    return NoiseMap(
        data=data,
        width=img.width,
        height=img.height,
        seed=0
    )


def save_point_set(point_set: PointSet, path: str) -> None:
    data = {
        "width": point_set.width,
        "height": point_set.height,
        "seed": point_set.seed,
        "points": point_set.points.tolist()
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_point_set(path: str) -> PointSet:
    with open(path, 'r') as f:
        data = json.load(f)
    return PointSet(
        points=np.array(data["points"], dtype=np.float32),
        width=data["width"],
        height=data["height"],
        seed=data["seed"]
    )