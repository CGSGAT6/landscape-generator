from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.indices)

    def to_interleaved(self) -> np.ndarray:
        return np.column_stack([self.vertices, self.normals, self.uvs])
