from __future__ import annotations

import moderngl
import numpy as np
from PIL import Image

import pyrr

from renderer.renderer import RenderEngine
from renderer.model import Model
from renderer.texture import Texture
from landscape import Landscape


class RenderBridge:
    def __init__(self, width: int, height: int):
        self.ctx = moderngl.create_standalone_context(require=430)
        self.renderer = RenderEngine()
        self.renderer.create_from_window(self.ctx)
        self.renderer.init()

        self._landscape: Landscape | None = None
        self._grid_model: Model | None = None
        self._flat_model: Model | None = None
        self._biome_tex: Texture | None = None
        self._show_flat: bool = False

        self._resize_fbo(width, height)
        self.renderer.set_screen_fbo(self.fbo)

    def _resize_fbo(self, w: int, h: int) -> None:
        self._color_tex = self.ctx.texture((w, h), components=4, dtype="f1")
        self.depth = self.ctx.depth_texture((w, h))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self._color_tex],
            depth_attachment=self.depth,
        )

    def resize(self, w: int, h: int) -> None:
        self._resize_fbo(w, h)
        self.ctx.gc()
        self.renderer.set_screen_fbo(self.fbo)
        self.renderer.resize(w, h)

    def set_landscape(self, landscape: Landscape | None, from_file: bool = False) -> None:
        self._landscape = landscape
        if self._grid_model is not None:
            self._grid_model.clear_prims()
        if self._flat_model is not None:
            self._flat_model.clear_prims()
        self._grid_model = None
        self._flat_model = None
        self._biome_tex = None
        self.ctx.gc()

        if landscape is None:
            return

        if from_file:
            if landscape.grid_mesh_path is not None:
                self._grid_model = self.renderer.create_model(file_path=landscape.grid_mesh_path)
            if landscape.flat_mesh_path is not None:
                self._flat_model = self.renderer.create_model(file_path=landscape.flat_mesh_path)
        else:
            if landscape.grid_mesh is not None:
                verts = landscape.grid_mesh.to_interleaved()
                prim = self.renderer.create_prim(
                    vertices=verts,
                    indeces=landscape.grid_mesh.indices,
                )
                self._grid_model = self.renderer.create_model(prims=[prim])
            if landscape.flat_mesh is not None:
                verts = landscape.flat_mesh.to_interleaved()
                prim = self.renderer.create_prim(
                    vertices=verts,
                    indeces=landscape.flat_mesh.indices,
                )
                self._flat_model = self.renderer.create_model(prims=[prim])
            if landscape.texture_image is not None:
                tex_data = np.array(landscape.texture_image.convert("RGBA"))
                self._biome_tex = self.renderer.create_texture(data=tex_data)
                for model in (self._grid_model, self._flat_model):
                    if model is not None:
                        for prim, _ in model.prims:
                            prim.mtl.set_texture(self._biome_tex, 0)

    def set_view_flat(self, flat: bool) -> None:
        self._show_flat = flat

    def render(self, time: float, dt: float) -> Image.Image:
        self.fbo.use()
        self.renderer.begin_frame(time, dt)

        model = self._flat_model if self._show_flat else self._grid_model
        if model is not None:
            center = (model.bbox_max + model.bbox_min) / 2
            model.render(pyrr.Matrix44.from_translation(-center))

        self.renderer.end_frame()

        w, h = self.fbo.size
        pixels = self.fbo.read(components=4, alignment=1)
        img = Image.frombytes("RGBA", (w, h), pixels)
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    def release(self) -> None:
        if hasattr(self, "renderer"):
            self.renderer.close()
        self._landscape = None
        self._grid_model = None
        self._flat_model = None
        self._biome_tex = None
        self._color_tex = None
        self.fbo = None
        self.depth = None
