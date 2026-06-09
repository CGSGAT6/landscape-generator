from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from geometry import Mesh, save_obj, save_mtl


@dataclass
class Landscape:
    height_map: np.ndarray
    biome_map: np.ndarray
    texture_path: Path | None = None
    texture_image: Image.Image | None = None
    grid_mesh: Mesh | None = None
    grid_mesh_path: Path | None = None
    flat_mesh: Mesh | None = None
    flat_mesh_path: Path | None = None
    sea_level: float = 0.5
    tree_positions: np.ndarray | None = None
    tree_model_path: Path | None = None
    seed_height: int = 0
    seed_moisture: int = 0
    seed_trees: int = 0

    _FORMAT_VERSION: ClassVar[int] = 1

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        tex_dir = root / "textures"
        mesh_dir = root / "meshes"
        model_dir = root / "models"
        data_dir = root / "data"

        tex_dir.mkdir(parents=True, exist_ok=True)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict = {
            "format_version": self._FORMAT_VERSION,
            "seed_height": self.seed_height,
            "seed_moisture": self.seed_moisture,
            "seed_trees": self.seed_trees,
            "sea_level": self.sea_level,
        }

        rel = root.resolve().parent

        def relpath(target: Path) -> str:
            return str(target.resolve().relative_to(rel).as_posix())

        np.save(data_dir / "height_map.npy", self.height_map)
        manifest["height_map"] = relpath(data_dir / "height_map.npy")

        np.save(data_dir / "biome_map.npy", self.biome_map)
        manifest["biome_map"] = relpath(data_dir / "biome_map.npy")

        if self.texture_path is not None:
            dst = tex_dir / self.texture_path.name
            shutil.copy2(self.texture_path, dst)
            manifest["texture"] = relpath(dst)

        if self.grid_mesh is not None:
            obj_path = mesh_dir / "grid.obj"
            mtl_path = mesh_dir / "grid.mtl"
            save_obj(self.grid_mesh, obj_path,
                     material_name="grid",
                     mtl_filename="grid.mtl")
            tex_rel = None
            if self.texture_path is not None:
                tex_rel = os.path.relpath(
                    (tex_dir / self.texture_path.name).resolve(),
                    mesh_dir.resolve(),
                ).replace("\\", "/")
            save_mtl(mtl_path, material_name="grid",
                     texture_rel_path=tex_rel)
            manifest["grid_mesh"] = relpath(obj_path)
            manifest["grid_mtl"] = relpath(mtl_path)
            self.grid_mesh_path = obj_path.resolve()

        if self.flat_mesh is not None:
            obj_path = mesh_dir / "flat.obj"
            mtl_path = mesh_dir / "flat.mtl"
            save_obj(self.flat_mesh, obj_path,
                     material_name="flat",
                     mtl_filename="flat.mtl")
            tex_rel = None
            if self.texture_path is not None:
                tex_rel = os.path.relpath(
                    (tex_dir / self.texture_path.name).resolve(),
                    mesh_dir.resolve(),
                ).replace("\\", "/")
            save_mtl(mtl_path, material_name="flat",
                     texture_rel_path=tex_rel)
            manifest["flat_mesh"] = relpath(obj_path)
            manifest["flat_mtl"] = relpath(mtl_path)
            self.flat_mesh_path = obj_path.resolve()

        if self.tree_positions is not None:
            np.save(data_dir / "tree_positions.npy", self.tree_positions)
            manifest["tree_positions"] = relpath(data_dir / "tree_positions.npy")

        if self.tree_model_path is not None:
            dst = model_dir / self.tree_model_path.name
            shutil.copy2(self.tree_model_path, dst)
            manifest["tree_model"] = relpath(dst)

        with open(root / "landscape.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Landscape:
        root = Path(path)
        with open(root / "landscape.json", encoding="utf-8") as f:
            manifest = json.load(f)

        def abs_path(rel: str) -> Path:
            return (root.parent / rel).resolve()

        height_map = np.load(abs_path(manifest["height_map"]))
        biome_map = np.load(abs_path(manifest["biome_map"]))

        texture_path: Path | None = None
        texture_image: Image.Image | None = None
        if "texture" in manifest:
            texture_path = abs_path(manifest["texture"])
            texture_image = Image.open(texture_path)

        grid_mesh: Mesh | None = None
        grid_mesh_path: Path | None = None
        if "grid_mesh" in manifest:
            from geometry import load_obj
            grid_mesh_path = abs_path(manifest["grid_mesh"])
            grid_mesh = load_obj(str(grid_mesh_path))

        flat_mesh: Mesh | None = None
        flat_mesh_path: Path | None = None
        if "flat_mesh" in manifest:
            from geometry import load_obj
            flat_mesh_path = abs_path(manifest["flat_mesh"])
            flat_mesh = load_obj(str(flat_mesh_path))

        tree_positions: np.ndarray | None = None
        if "tree_positions" in manifest:
            tree_positions = np.load(abs_path(manifest["tree_positions"]))

        tree_model_path: Path | None = None
        if "tree_model" in manifest:
            tree_model_path = abs_path(manifest["tree_model"])

        return cls(
            height_map=height_map,
            biome_map=biome_map,
            texture_path=texture_path,
            texture_image=texture_image,
            grid_mesh=grid_mesh,
            grid_mesh_path=grid_mesh_path,
            flat_mesh=flat_mesh,
            flat_mesh_path=flat_mesh_path,
            sea_level=manifest.get("sea_level", 0.5),
            tree_positions=tree_positions,
            tree_model_path=tree_model_path,
            seed_height=manifest.get("seed_height", 0),
            seed_moisture=manifest.get("seed_moisture", 0),
            seed_trees=manifest.get("seed_trees", 0),
        )
