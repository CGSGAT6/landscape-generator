from __future__ import annotations

import numpy as np


class Selector:
    """Двухуровневый селектор: selector[height][moisture] -> layer[moisture] -> biome.

    Правило: insert(threshold, value) — для ключа x <= threshold возвращается value.
    Пороги проверяются от наименьшего к наибольшему (самый специфичный — первый).
    """

    def __init__(self, default=None):
        self._points: list[tuple[float, object]] = []
        self._default = default

    def insert(self, threshold: float, value: object) -> Selector:
        self._points.append((threshold, value))
        self._points.sort(key=lambda x: x[0])
        return self

    def __getitem__(self, key: float) -> object:
        for threshold, value in self._points:
            if key <= threshold:
                return value
        return self._default

    def __repr__(self) -> str:
        return f"Selector(default={self._default!r}, points={self._points!r})"


def apply_selector_to_arrays(selector: Selector,
                              height_2d: np.ndarray,
                              moisture_2d: np.ndarray) -> np.ndarray:
    H, W = height_2d.shape
    result = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            layer = selector[height_2d[y, x]]
            if isinstance(layer, Selector):
                result[y, x] = layer[moisture_2d[y, x]]
            else:
                result[y, x] = layer
    return result


def _make_default_selector(
    sea_level: float = 0.20,
) -> Selector:
    """Селектор по умолчанию: высота + влажность → биом.

    Высота   | Влажность         | Биом
    --------|-------------------|------
    ≤0.20   | любая             | WATER
    0.20–35 | любая             | SAND
    0.35–55 | ≤0.4→GRASS >0.4→FOREST
    0.55–75 | ≤0.3→GRASS >0.3→FOREST
    0.75–90 | ≤0.2→ROCK ≤0.5→GRASS >0.5→FOREST
    >0.90   | ≤0.3→ROCK >0.3→SNOW
    """
    from .biome import BiomeType

    low_moisture = (
        Selector(BiomeType.FOREST)
        .insert(0.4, BiomeType.GRASS)
    )
    mid_moisture = (
        Selector(BiomeType.FOREST)
        .insert(0.3, BiomeType.GRASS)
    )
    high_moisture = (
        Selector(BiomeType.FOREST)
        .insert(0.5, BiomeType.GRASS)
        .insert(0.2, BiomeType.ROCK)
    )
    peak_moisture = (
        Selector(BiomeType.SNOW)
        .insert(0.3, BiomeType.ROCK)
    )

    height_sel: Selector = (
        Selector(peak_moisture)          # > 0.90
        .insert(0.90, high_moisture)     # 0.75–0.90
        .insert(0.75, mid_moisture)      # 0.55–0.75
        .insert(0.55, low_moisture)      # 0.35–0.55
        .insert(0.35, Selector(BiomeType.SAND))
        .insert(sea_level, Selector(BiomeType.WATER))
    )
    return height_sel
