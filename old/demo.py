import sys
import os
import time

import numpy as np
import dearpygui.dearpygui as dpg
import moderngl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from renderer import (
    RendererContext, Camera, Model, Texture,
    ShaderProgram, Scene, Framebuffer, RenderSettings,
)

WIDTH, HEIGHT = 800, 600


def _make_cube():
    p = 1.0
    data = np.array([
        [-p, -p,  p,  0, 0, 1,  0, 0],
        [ p, -p,  p,  0, 0, 1,  1, 0],
        [ p,  p,  p,  0, 0, 1,  1, 1],
        [-p,  p,  p,  0, 0, 1,  0, 1],

        [ p, -p, -p,  0, 0,-1,  1, 0],
        [-p, -p, -p,  0, 0,-1,  0, 0],
        [-p,  p, -p,  0, 0,-1,  0, 1],
        [ p,  p, -p,  0, 0,-1,  1, 1],

        [-p,  p, -p,  0, 1, 0,  0, 0],
        [-p,  p,  p,  0, 1, 0,  0, 1],
        [ p,  p,  p,  0, 1, 0,  1, 1],
        [ p,  p, -p,  0, 1, 0,  1, 0],

        [-p, -p,  p,  0,-1, 0,  0, 0],
        [ p, -p,  p,  0,-1, 0,  1, 0],
        [ p, -p, -p,  0,-1, 0,  1, 1],
        [-p, -p, -p,  0,-1, 0,  0, 1],

        [ p, -p, -p,  1, 0, 0,  0, 0],
        [ p, -p,  p,  1, 0, 0,  0, 1],
        [ p,  p,  p,  1, 0, 0,  1, 1],
        [ p,  p, -p,  1, 0, 0,  1, 0],

        [-p, -p,  p, -1, 0, 0,  0, 0],
        [-p, -p, -p, -1, 0, 0,  1, 0],
        [-p,  p, -p, -1, 0, 0,  1, 1],
        [-p,  p,  p, -1, 0, 0,  0, 1],
    ], dtype=np.float32)

    vertices = data[:, :3].copy()
    normals = data[:, 3:6].copy()
    uvs = data[:, 6:8].copy()
    indices = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [8, 9, 10], [8, 10, 11],
        [12, 13, 14], [12, 14, 15],
        [16, 17, 18], [16, 18, 19],
        [20, 21, 22], [20, 22, 23],
    ], dtype=np.uint32)

    return vertices, indices, normals, uvs


def _make_checkerboard(size: int = 64) -> np.ndarray:
    tex = np.ones((size, size, 4), dtype=np.uint8) * 255
    checker = (np.arange(size)[:, None] // 8 + np.arange(size)[None, :] // 8) % 2 == 0
    tex[checker] = [200, 50, 50, 255]
    tex[~checker] = [50, 50, 200, 255]
    return tex


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = -s
    m[2, 0] = s
    m[2, 2] = c
    return m


def demo():
    dpg.create_context()
    dpg.create_viewport(title='Renderer Demo', width=WIDTH, height=HEIGHT)
    dpg.setup_dearpygui()

    with dpg.texture_registry():
        dpg.add_dynamic_texture(
            WIDTH, HEIGHT, [0.0] * WIDTH * HEIGHT * 4,
            tag='render',
        )

    with dpg.window(label='3D Viewport', no_close=True, no_collapse=True,
                    width=WIDTH, height=HEIGHT):
        dpg.add_image('render')

    dpg.show_viewport()

    rctx = RendererContext()
    rctx.create_standalone(WIDTH, HEIGHT)
    ctx = rctx.ctx

    camera = Camera(aspect=WIDTH / HEIGHT)
    camera.set_position(3.5, 2.5, 4.0)
    camera.set_target(0, 0, 0)

    shader_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'shaders')
    shader = ShaderProgram(
        ctx,
        os.path.join(shader_dir, 'simple.vert'),
        os.path.join(shader_dir, 'simple.frag'),
    )

    verts, idxs, norms, uvs = _make_cube()
    model = Model(ctx)
    model.load_from_mesh(verts, idxs, norms, uvs)

    tex_data = _make_checkerboard()
    texture = Texture(ctx)
    texture.load_numpy(tex_data)

    scene = Scene(ctx)
    scene.add_model('cube', model)
    scene.set_directional_light((0.5, -0.8, -0.3), (1, 1, 1), 1.0)

    fbo = Framebuffer(ctx, WIDTH, HEIGHT)

    settings = RenderSettings()
    start_time = time.time()

    def resize_cb(sender, app_data):
        global WIDTH, HEIGHT
        w, h = app_data['width'], app_data['height']
        WIDTH, HEIGHT = w, h
        rctx.resize(w, h)
        camera.set_aspect(w, h)
        fbo.resize(w, h)
        dpg.configure_item('render', width=w, height=h)

    dpg.set_viewport_resize_callback(resize_cb)

    while dpg.is_dearpygui_running():
        angle = time.time() - start_time
        scene.set_model_transform('cube', _rotation_y(angle * 0.5))

        fbo.bind()
        texture.bind(0)

        fbo.clear(*settings.background_color)

        scene.render(
            camera, shader, settings,
            extra_uniforms={'texture0': 0},
        )

        pixels = fbo.read(components=4, dtype='u1')
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
        img = np.flipud(img).astype(np.float32) / 255.0
        dpg.set_value('render', img.ravel().tolist())

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == '__main__':
    demo()
