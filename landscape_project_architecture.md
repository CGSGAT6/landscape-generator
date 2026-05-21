# Архитектура проекта: Генерация и визуализация ландшафтов

## Обзор

Проект разбит на **4 независимых первичных модуля**, которые разрабатываются параллельно, и **2 вторичных модуля**, объединяющих их в готовое приложение.

### Выбор UI-фреймворка: Dear PyGui

**Рекомендация: Dear PyGui** — оптимальный выбор для данного проекта по следующим причинам:

- Работает в **immediate mode** (как ImGui) — легко встраивается в render loop ModernGL
- **Нативная интеграция с OpenGL** — можно передавать GL-текстуры напрямую в виджеты
- Простое добавление интерактивных элементов (слайдеры, кнопки) прямо в viewport
- Не требует сложной настройки контекста в отличие от PyQt6

**Установка:** `pip install dearpygui`

---

## Первичные модули

---

### Модуль 1: `noise` — Генерация шума

**Разработчик:** [имя]  
**Зависимости:** `numpy`, `Pillow`  
**Внешние зависимости от других модулей:** нет

#### Структура файлов

```
noise/
├── __init__.py
├── perlin.py        # Шум Перлина (1D, 2D, 3D)
├── fractal.py       # Фрактальный шум, октавы, red noise
├── sampling.py      # Генерация псевдослучайных точек
├── filters.py       # Постобработка шума
├── io.py            # Сохранение/загрузка
└── types.py         # Общие типы данных
```

#### Типы данных (`types.py`)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class NoiseMap:
    """Двумерная карта шума."""
    data: np.ndarray        # shape (H, W), dtype float32, диапазон [0.0, 1.0]
    width: int
    height: int
    seed: int

@dataclass
class PointSet:
    """Набор двумерных точек."""
    points: np.ndarray      # shape (N, 2), dtype float32
    width: float
    height: float
    seed: int
```

#### Класс `PerlinNoise` (`perlin.py`)

```python
class PerlinNoise:
    def __init__(self, seed: int = 0):
        """Инициализация с таблицей перестановок по seed."""
        ...

    def noise1d(self, x: float) -> float:
        """Шум Перлина в одной точке 1D."""
        ...

    def noise2d(self, x: float, y: float) -> float:
        """Шум Перлина в одной точке 2D."""
        ...

    def noise3d(self, x: float, y: float, z: float) -> float:
        """Шум Перлина в одной точке 3D."""
        ...

    def generate2d(self, width: int, height: int,
                   scale: float = 1.0) -> NoiseMap:
        """Генерация карты шума 2D через numpy (векторизованно)."""
        ...
```

#### Класс `FractalNoise` (`fractal.py`)

```python
class FractalNoise:
    def __init__(self, base_noise: PerlinNoise):
        ...

    def fbm(self, width: int, height: int,
            octaves: int = 6,
            lacunarity: float = 2.0,
            gain: float = 0.5,
            scale: float = 1.0) -> NoiseMap:
        """Fractional Brownian Motion — классический фрактальный шум."""
        ...

    def ridged(self, width: int, height: int,
               octaves: int = 6, **kwargs) -> NoiseMap:
        """Ridged multifractal — хребты и горные гряды."""
        ...

    def red_noise(self, width: int, height: int,
                  exponent: float = 2.0) -> NoiseMap:
        """Red noise (1/f^n) через FFT."""
        ...
```

#### Класс `PoissonSampler` (`sampling.py`)

```python
class PoissonSampler:
    def __init__(self, seed: int = 0):
        ...

    def sample(self, width: float, height: float,
               min_distance: float,
               max_attempts: int = 30) -> PointSet:
        """
        Poisson Disk Sampling — равномерно распределённые точки
        без кластеризации. Алгоритм Bridson.
        """
        ...

    def uniform_random(self, width: float, height: float,
                       count: int) -> PointSet:
        """Простая равномерная случайная выборка."""
        ...
```

#### Класс `NoiseFilter` (`filters.py`)

```python
class NoiseFilter:
    @staticmethod
    def power_curve(noise_map: NoiseMap, exponent: float) -> NoiseMap:
        """Степенная функция: x^exponent. Усиливает контраст."""
        ...

    @staticmethod
    def step(noise_map: NoiseMap, levels: int) -> NoiseMap:
        """Ступенчатая функция — квантизация значений."""
        ...

    @staticmethod
    def remap(noise_map: NoiseMap,
              in_min: float, in_max: float,
              out_min: float, out_max: float) -> NoiseMap:
        """Линейный remapping диапазона значений."""
        ...

    @staticmethod
    def apply_curve(noise_map: NoiseMap,
                    control_points: list[tuple[float, float]]) -> NoiseMap:
        """Произвольная кривая через контрольные точки (сплайн)."""
        ...
```

#### Функции ввода-вывода (`io.py`)

```python
def save_noise_as_image(noise_map: NoiseMap, path: str,
                        colormap: str = "grayscale") -> None:
    """Сохранить NoiseMap как PNG/BMP. colormap: 'grayscale', 'terrain', 'heat'."""
    ...

def save_point_set(point_set: PointSet, path: str) -> None:
    """Сохранить PointSet в формат .pts (JSON-based)."""
    ...

def load_point_set(path: str) -> PointSet:
    """Загрузить PointSet из .pts файла."""
    ...

def load_noise_from_image(path: str) -> NoiseMap:
    """Загрузить изображение как NoiseMap (grayscale)."""
    ...
```

#### Публичное API (`__init__.py`)

```python
from .perlin import PerlinNoise
from .fractal import FractalNoise
from .sampling import PoissonSampler
from .filters import NoiseFilter
from .io import save_noise_as_image, save_point_set, load_point_set, load_noise_from_image
from .types import NoiseMap, PointSet
```

#### Пример использования

```python
import noise

gen = noise.PerlinNoise(seed=42)
fractal = noise.FractalNoise(gen)

hmap = fractal.fbm(512, 512, octaves=8)
hmap = noise.NoiseFilter.power_curve(hmap, exponent=1.5)

noise.save_noise_as_image(hmap, "heightmap.png", colormap="terrain")
```

---

### Модуль 2: `app_base` — Базовое приложение

**Разработчик:** [имя]  
**Зависимости:** `dearpygui`, `Pillow`, `numpy`  
**Внешние зависимости от других модулей:** нет (принимает данные через интерфейс)

#### Структура файлов

```
app_base/
├── __init__.py
├── application.py   # Главный класс приложения
├── window.py        # Управление окнами Dear PyGui
├── image_viewer.py  # Просмотр изображений
├── input_handler.py # Обработка ввода
├── render_bridge.py # Интерфейс для передачи контекста рендеру
└── types.py         # Общие типы
```

#### Типы данных (`types.py`)

```python
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class RenderContext:
    """Данные, передаваемые системе рендера."""
    viewport_width: int
    viewport_height: int
    gl_texture_id: int | None = None   # OpenGL texture handle
    camera_position: tuple = (0, 5, 10)
    camera_target: tuple = (0, 0, 0)
    extra: dict = field(default_factory=dict)

@dataclass
class AppConfig:
    title: str = "Landscape Generator"
    width: int = 1280
    height: int = 720
    target_fps: int = 60
```

#### Класс `Application` (`application.py`)

```python
class Application:
    def __init__(self, config: AppConfig = AppConfig()):
        ...

    def set_render_callback(self, callback: Callable[[RenderContext], None]) -> None:
        """
        Зарегистрировать callback для рендеринга.
        Вызывается каждый кадр, если рендер активен.
        """
        ...

    def set_image_viewer_source(self, image_data: np.ndarray | str) -> None:
        """Показать изображение (numpy array или путь к файлу) в просмотрщике."""
        ...

    def get_render_context(self) -> RenderContext:
        """Получить текущий контекст для передачи рендеру."""
        ...

    def add_menu_item(self, menu: str, label: str,
                      callback: Callable) -> None:
        """Добавить пункт меню."""
        ...

    def run(self) -> None:
        """Запустить главный цикл приложения."""
        ...

    def stop(self) -> None:
        """Завершить приложение."""
        ...
```

#### Класс `ImageViewer` (`image_viewer.py`)

```python
class ImageViewer:
    def __init__(self, parent_tag: str):
        ...

    def show_numpy(self, array: np.ndarray, label: str = "") -> None:
        """Отобразить numpy array как изображение (RGB или grayscale)."""
        ...

    def show_file(self, path: str) -> None:
        """Загрузить и отобразить изображение из файла."""
        ...

    def clear(self) -> None:
        ...

    def set_zoom(self, factor: float) -> None:
        ...
```

#### Класс `RenderBridge` (`render_bridge.py`)

```python
class RenderBridge:
    """
    Управляет передачей Dear PyGui viewport во внешний рендерер.
    Создаёт нативный GL-контекст, совместимый с ModernGL.
    """

    def __init__(self):
        ...

    def create_viewport(self, width: int, height: int) -> int:
        """
        Создать Dear PyGui viewport для рендера.
        Возвращает тег viewport.
        """
        ...

    def get_gl_context_info(self) -> dict:
        """
        Вернуть информацию об OpenGL-контексте
        (нужна для инициализации ModernGL).
        """
        ...

    def resize(self, width: int, height: int) -> None:
        ...
```

#### Публичное API

```python
from .application import Application
from .types import AppConfig, RenderContext
from .image_viewer import ImageViewer
from .render_bridge import RenderBridge
```

---

### Модуль 3: `geometry` — Генерация геометрии

**Разработчик:** [имя]  
**Зависимости:** `numpy`, `scipy` (триангуляция Делоне)  
**Внешние зависимости от других модулей:** нет (принимает `np.ndarray` и списки точек)

#### Структура файлов

```
geometry/
├── __init__.py
├── mesh.py          # Базовые структуры данных
├── grid.py          # Построение Grid из HeightMap
├── delaunay.py      # Триангуляция Делоне
├── io.py            # Экспорт в OBJ и другие форматы
└── utils.py         # Вспомогательные функции (нормали, UV и т.д.)
```

#### Типы данных (`mesh.py`)

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Mesh:
    """Универсальный полигональный меш."""
    vertices: np.ndarray    # shape (N, 3), float32 — позиции XYZ
    indices: np.ndarray     # shape (M, 3), uint32 — треугольники
    normals: np.ndarray     # shape (N, 3), float32 — нормали
    uvs: np.ndarray         # shape (N, 2), float32 — текстурные координаты

    @property
    def vertex_count(self) -> int: ...

    @property
    def face_count(self) -> int: ...
```

#### Класс `GridBuilder` (`grid.py`)

```python
class GridBuilder:
    """Строит регулярную сетку (Grid) из карты высот."""

    def build(self, heightmap: np.ndarray,
              x_scale: float = 1.0,
              y_scale: float = 1.0,
              z_scale: float = 1.0) -> Mesh:
        """
        Построить меш из двумерного массива высот.
        heightmap: np.ndarray shape (H, W), float32, [0.0, 1.0]
        x_scale, y_scale: горизонтальный масштаб
        z_scale: вертикальный масштаб (высота)
        """
        ...

    def build_with_lod(self, heightmap: np.ndarray,
                       levels: int = 3, **kwargs) -> list[Mesh]:
        """
        Построить несколько уровней детализации (LOD).
        levels=1 — исходное, levels=2 — в 2 раза меньше и т.д.
        """
        ...
```

#### Класс `DelaunayTriangulator` (`delaunay.py`)

```python
class DelaunayTriangulator:
    """Триангуляция Делоне по набору точек (через scipy.spatial.Delaunay)."""

    def triangulate(self, points: np.ndarray,
                    heights: np.ndarray | None = None) -> Mesh:
        """
        points: np.ndarray shape (N, 2) — координаты XZ
        heights: np.ndarray shape (N,) — высоты Y (или 0, если None)
        Возвращает меш с нормалями и UV.
        """
        ...

    def triangulate_with_constraints(self, points: np.ndarray,
                                     heights: np.ndarray | None = None,
                                     boundary: np.ndarray | None = None) -> Mesh:
        """
        Триангуляция с ограничениями (constrained Delaunay).
        boundary: np.ndarray shape (K, 2) — контурные точки границы
        """
        ...
```

#### Функции ввода-вывода (`io.py`)

```python
def save_obj(mesh: Mesh, path: str,
             material_name: str | None = None) -> None:
    """Экспорт меша в формат Wavefront OBJ (+ MTL если задан материал)."""
    ...

def load_obj(path: str) -> Mesh:
    """Загрузить меш из OBJ файла."""
    ...

def save_ply(mesh: Mesh, path: str) -> None:
    """Экспорт в Stanford PLY (бинарный, эффективнее OBJ)."""
    ...
```

#### Вспомогательные функции (`utils.py`)

```python
def compute_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Вычислить нормали для каждой вершины (усреднение по граням)."""
    ...

def compute_uvs_planar(vertices: np.ndarray) -> np.ndarray:
    """UV-развёртка через плоскую проекцию по XZ."""
    ...

def merge_meshes(meshes: list[Mesh]) -> Mesh:
    """Объединить несколько мешей в один."""
    ...
```

#### Публичное API

```python
from .mesh import Mesh
from .grid import GridBuilder
from .delaunay import DelaunayTriangulator
from .io import save_obj, load_obj, save_ply
from .utils import compute_normals, compute_uvs_planar, merge_meshes
```

#### Пример использования

```python
import numpy as np
import geometry

heightmap = np.random.rand(256, 256).astype(np.float32)

builder = geometry.GridBuilder()
mesh = builder.build(heightmap, x_scale=1.0, z_scale=50.0)

geometry.save_obj(mesh, "terrain.obj")
```

---

### Модуль 4: `renderer` — Рендер (ModernGL)

**Разработчик:** [имя]  
**Зависимости:** `moderngl`, `numpy`, `Pillow`, `pyrr` (матрицы)  
**Внешние зависимости от других модулей:** нет (принимает стандартные numpy-данные)

#### Структура файлов

```
renderer/
├── __init__.py
├── context.py       # Инициализация ModernGL контекста
├── camera.py        # Камера и управление
├── scene.py         # Сцена: объекты, источники света
├── model.py         # Загрузка и хранение моделей
├── texture.py       # Загрузка и управление текстурами
├── shader.py        # Управление шейдерами (с hot-reload)
├── framebuffer.py   # Фреймбуферы для постобработки
└── types.py         # Типы данных
```

#### Типы данных (`types.py`)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class RenderSettings:
    background_color: tuple = (0.1, 0.1, 0.15, 1.0)
    wireframe: bool = False
    enable_fog: bool = True
    fog_density: float = 0.02
```

#### Класс `RendererContext` (`context.py`)

```python
class RendererContext:
    """Обёртка над ModernGL контекстом."""

    def __init__(self):
        self.ctx: moderngl.Context | None = None

    def create_standalone(self, width: int, height: int) -> None:
        """Создать автономный контекст (для headless рендеринга)."""
        ...

    def create_from_window(self, window_handle) -> None:
        """
        Подключиться к существующему GL-контексту окна
        (Dear PyGui или другого оконного менеджера).
        """
        ...

    def resize(self, width: int, height: int) -> None:
        ...

    def clear(self, color: tuple | None = None) -> None:
        ...
```

#### Класс `Camera` (`camera.py`)

```python
class Camera:
    def __init__(self, fov: float = 60.0,
                 aspect: float = 16/9,
                 near: float = 0.1,
                 far: float = 10000.0):
        ...

    # Позиция и ориентация
    def set_position(self, x: float, y: float, z: float) -> None: ...
    def set_target(self, x: float, y: float, z: float) -> None: ...

    # Управление orbit-камерой (вращение вокруг точки)
    def orbit(self, delta_yaw: float, delta_pitch: float) -> None: ...
    def zoom(self, delta: float) -> None: ...
    def pan(self, delta_x: float, delta_y: float) -> None: ...

    # Матрицы
    def get_view_matrix(self) -> np.ndarray: ...         # 4x4 float32
    def get_projection_matrix(self) -> np.ndarray: ...   # 4x4 float32
    def get_vp_matrix(self) -> np.ndarray: ...           # view * projection
```

#### Класс `ShaderProgram` (`shader.py`)

```python
class ShaderProgram:
    def __init__(self, ctx: RendererContext,
                 vert_path: str, frag_path: str):
        ...

    def reload(self) -> bool:
        """
        Перекомпилировать шейдеры из файлов на лету.
        Возвращает True при успехе, False при ошибке компиляции
        (старая программа остаётся активной).
        """
        ...

    def watch_and_reload(self, interval: float = 0.5) -> None:
        """Запустить фоновый поток, следящий за изменением файлов."""
        ...

    def use(self) -> None: ...

    def set_uniform(self, name: str, value) -> None:
        """Установить uniform (поддерживает int, float, vec, mat)."""
        ...
```

#### Класс `Model` (`model.py`)

```python
class Model:
    def __init__(self, ctx: RendererContext):
        ...

    def load_obj(self, path: str) -> None:
        """Загрузить OBJ и создать VAO/VBO в GPU."""
        ...

    def load_from_mesh(self, vertices: np.ndarray,
                       indices: np.ndarray,
                       normals: np.ndarray,
                       uvs: np.ndarray) -> None:
        """Загрузить данные напрямую из numpy-массивов."""
        ...

    def update_vertices(self, vertices: np.ndarray) -> None:
        """Обновить позиции вершин без пересоздания VAO."""
        ...

    def draw(self, shader: ShaderProgram) -> None: ...
```

#### Класс `Texture` (`texture.py`)

```python
class Texture:
    def __init__(self, ctx: RendererContext):
        ...

    def load_file(self, path: str) -> None:
        """Загрузить текстуру из файла (PNG, JPG)."""
        ...

    def load_numpy(self, data: np.ndarray) -> None:
        """Загрузить текстуру из numpy array (H, W, C)."""
        ...

    def bind(self, slot: int = 0) -> None: ...

    @property
    def gl_handle(self) -> int:
        """OpenGL texture ID (для передачи в Dear PyGui)."""
        ...
```

#### Класс `Framebuffer` (`framebuffer.py`)

```python
class Framebuffer:
    def __init__(self, ctx: RendererContext,
                 width: int, height: int):
        ...

    def bind(self) -> None:
        """Переключить рендеринг в этот фреймбуфер."""
        ...

    def unbind(self) -> None:
        """Вернуться к основному фреймбуферу."""
        ...

    def get_color_texture(self) -> Texture:
        """Получить цветовую текстуру для постобработки."""
        ...

    def resize(self, width: int, height: int) -> None: ...
```

#### Класс `Scene` (`scene.py`)

```python
class Scene:
    def __init__(self, ctx: RendererContext):
        ...

    def add_model(self, name: str, model: Model) -> None: ...
    def remove_model(self, name: str) -> None: ...
    def get_model(self, name: str) -> Model | None: ...

    def set_directional_light(self, direction: tuple,
                              color: tuple = (1, 1, 1),
                              intensity: float = 1.0) -> None: ...

    def render(self, camera: Camera,
               shader: ShaderProgram,
               settings: RenderSettings = RenderSettings()) -> None:
        """Отрендерить всю сцену."""
        ...
```

#### Публичное API

```python
from .context import RendererContext
from .camera import Camera
from .scene import Scene
from .model import Model
from .texture import Texture
from .shader import ShaderProgram
from .framebuffer import Framebuffer
from .types import RenderSettings
```

---

## Вторичные модули

---

### Модуль 5: `landscape` — Генерация ландшафта

**Зависимости:** `noise`, `geometry`  
**Разработчик:** объединяет работу 2 человек

#### Ключевые классы

```python
@dataclass
class LandscapeData:
    heightmap: np.ndarray        # (H, W) float32
    biome_map: np.ndarray        # (H, W) uint8 — индекс биома
    mesh: Mesh | None
    texture: np.ndarray | None   # (H, W, 4) RGBA
    metadata: dict               # seed, параметры и т.д.

class LandscapeGenerator:
    def generate(self, config: LandscapeConfig) -> LandscapeData: ...
    def save(self, data: LandscapeData, path: str) -> None: ...    # JSON + PNG
    def load(self, path: str) -> LandscapeData: ...
```

---

### Модуль 6: `app` — Финальное приложение

**Зависимости:** `app_base`, `landscape`, `renderer`  
**Разработчик:** объединяет всё

#### Ответственности

- UI для настройки параметров генератора
- Переключение между 2D-видом (biome map) и 3D-рендером
- Вызов `LandscapeGenerator` и передача результата в `renderer`

---

## Зависимости между модулями

```
noise ──────────────────────────────► landscape ──► app
geometry ───────────────────────────►             │
                                                   │
app_base ───────────────────────────────────────► app
renderer ───────────────────────────────────────►
```

**Первичные модули** (`noise`, `geometry`, `app_base`, `renderer`) — **полностью независимы** друг от друга. Каждый работает только со стандартными типами Python/NumPy.

---

## Общие соглашения

| Тема | Соглашение |
|------|------------|
| Координаты | Y — вверх, Z — вглубь (OpenGL стандарт) |
| Данные шума | `np.ndarray`, dtype `float32`, диапазон `[0.0, 1.0]` |
| Данные меша | `numpy` напрямую, никакого proprietary формата |
| Передача между модулями | Только через публичное API (`__init__.py`) |
| Тесты | Каждый модуль имеет `tests/` директорию |
| Форматы файлов | `.pts` (points), `.obj` (mesh), `.json` (landscape) |

---

## Стек технологий

| Компонент | Библиотека |
|-----------|-----------|
| UI | `dearpygui` |
| OpenGL | `moderngl` |
| Матрицы | `pyrr` |
| Изображения | `Pillow` |
| Вычисления | `numpy` |
| Триангуляция | `scipy.spatial.Delaunay` |

**Установка всего:**
```bash
pip install numpy scipy Pillow moderngl pyrr dearpygui
```
