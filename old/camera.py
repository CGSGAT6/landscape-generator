import numpy as np
from pyrr import Matrix44


class Camera:
    def __init__(self, fov: float = 60.0, aspect: float = 16 / 9,
                 near: float = 0.1, far: float = 10000.0):
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        self._distance = 10.0
        self._yaw = 0.0
        self._pitch = 30.0
        self._target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def set_position(self, x: float, y: float, z: float) -> None:
        offset = np.array([x, y, z], dtype=np.float32) - self._target
        self._distance = np.linalg.norm(offset)
        if self._distance < 1e-8:
            self._distance = 1e-8
        self._yaw = np.degrees(np.arctan2(offset[0], offset[2]))
        self._pitch = np.degrees(np.arcsin(np.clip(offset[1] / self._distance, -1.0, 1.0)))

    def set_target(self, x: float, y: float, z: float) -> None:
        self._target[:] = [x, y, z]

    def set_aspect(self, width: int | float, height: int | float) -> None:
        self.aspect = width / height

    def orbit(self, delta_yaw: float, delta_pitch: float) -> None:
        self._yaw += delta_yaw
        self._pitch = np.clip(self._pitch + delta_pitch, -89.0, 89.0)

    def zoom(self, delta: float) -> None:
        self._distance = max(0.1, self._distance - delta * 0.5)

    def pan(self, delta_x: float, delta_y: float) -> None:
        forward = self._get_forward()
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-8:
            right /= right_norm
        else:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = np.cross(right, forward)
        up_norm = np.linalg.norm(up)
        if up_norm > 1e-8:
            up /= up_norm
        self._target += right * delta_x + up * delta_y

    def _get_position(self) -> np.ndarray:
        yaw_rad = np.radians(self._yaw)
        pitch_rad = np.radians(self._pitch)
        x = self._distance * np.cos(pitch_rad) * np.sin(yaw_rad)
        y = self._distance * np.sin(pitch_rad)
        z = self._distance * np.cos(pitch_rad) * np.cos(yaw_rad)
        return self._target + np.array([x, y, z], dtype=np.float32)

    def _get_forward(self) -> np.ndarray:
        pos = self._get_position()
        d = self._target - pos
        norm = np.linalg.norm(d)
        if norm < 1e-8:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return d / norm

    def get_view_matrix(self) -> np.ndarray:
        pos = self._get_position()
        return np.array(Matrix44.look_at(pos, self._target, np.array([0.0, 1.0, 0.0], dtype=np.float32)), dtype=np.float32)

    def get_projection_matrix(self) -> np.ndarray:
        return np.array(Matrix44.perspective_projection(self.fov, self.aspect, self.near, self.far), dtype=np.float32)

    def get_vp_matrix(self) -> np.ndarray:
        return self.get_projection_matrix() @ self.get_view_matrix()
