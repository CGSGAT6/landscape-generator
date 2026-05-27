import numpy as np
from .types import PointSet


class PoissonSampler:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def sample(self, width: float, height: float,
               min_distance: float,
               max_attempts: int = 30) -> PointSet:
        cell_size = min_distance / np.sqrt(2)
        cols = int(np.ceil(width / cell_size))
        rows = int(np.ceil(height / cell_size))
        grid = np.full((rows, cols), -1, dtype=int)

        points = []
        active = []

        x0 = self.rng.uniform(0, width)
        y0 = self.rng.uniform(0, height)
        points.append([x0, y0])
        active.append(0)
        grid[int(y0 / cell_size), int(x0 / cell_size)] = 0

        while active:
            idx = self.rng.integers(len(active))
            px, py = points[active[idx]]
            found = False

            for _ in range(max_attempts):
                angle = self.rng.uniform(0, 2 * np.pi)
                radius = self.rng.uniform(min_distance, min_distance * 2)
                nx = px + np.cos(angle) * radius
                ny = py + np.sin(angle) * radius

                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue

                col = int(nx / cell_size)
                row = int(ny / cell_size)

                ok = True
                for i in range(max(0, row - 1), min(rows, row + 2)):
                    for j in range(max(0, col - 1), min(cols, col + 2)):
                        if grid[i, j] != -1:
                            qx, qy = points[grid[i, j]]
                            if (qx - nx) ** 2 + (qy - ny) ** 2 < min_distance ** 2:
                                ok = False
                                break
                    if not ok:
                        break

                if ok:
                    new_idx = len(points)
                    points.append([nx, ny])
                    active.append(new_idx)
                    grid[row, col] = new_idx
                    found = True
                    break

            if not found:
                active.pop(idx)

        return PointSet(
            points=np.array(points, dtype=np.float32),
            width=width,
            height=height,
            seed=self.seed
        )

    def uniform_random(self, width: float, height: float,
                       count: int) -> PointSet:
        points = self.rng.uniform(0, [width, height], size=(count, 2))
        return PointSet(
            points=points.astype(np.float32),
            width=width,
            height=height,
            seed=self.seed
        )