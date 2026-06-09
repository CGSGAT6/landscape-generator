"""Standalone (headless) renderer demo.
Creates a standalone ModernGL context (no window) and renders
with different primitive modes: TRIANGLES, TRIANGLE_STRIP, LINES, POINTS.
Each result is saved as a PNG image.
"""

import sys
import os

import numpy as np
import moderngl
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from renderer import (
    RendererContext, Camera, Model, Texture,
    ShaderProgram, Scene, Framebuffer,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIDTH, HEIGHT = 800, 600


def _make_cube():
    """Cube mesh -- 24 verts, 36 indices (12 triangles)."""
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


def _make_strip():
    """Triangle strip -- 4 verts, 4 indices => 2 triangles (a quad)."""
    vertices = np.array([
        [-1.0,  0.0,  1.0],
        [ 1.0,  0.0,  1.0],
        [-1.0,  0.0, -1.0],
        [ 1.0,  0.0, -1.0],
    ], dtype=np.float32)
    normals = np.tile([0.0, 1.0, 0.0], (4, 1)).astype(np.float32)
    uvs = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
    ], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.uint32)
    return vertices, indices, normals, uvs


def _make_lines():
    """Wireframe cube -- 8 verts, 24 indices (12 line segments)."""
    p = 1.0
    verts = np.array([
        [-p, -p, -p], [ p, -p, -p],
        [ p,  p, -p], [-p,  p, -p],
        [-p, -p,  p], [ p, -p,  p],
        [ p,  p,  p], [-p,  p,  p],
    ], dtype=np.float32)
    normals = np.tile([0.0, 1.0, 0.0], (8, 1)).astype(np.float32)
    uvs = np.zeros((8, 2), dtype=np.float32)
    indices = np.array([
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7,
    ], dtype=np.uint32)
    return verts, indices, normals, uvs


def _make_points():
    """Point cloud -- 500 random points inside a sphere."""
    rng = np.random.default_rng(42)
    n = 500
    phi = rng.uniform(0, 2 * np.pi, n)
    costheta = rng.uniform(-1, 1, n)
    theta = np.arccos(costheta)
    r = rng.uniform(0, 1.5, n) ** (1 / 3)

    verts = np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ]).astype(np.float32)
    normals = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    uvs = np.zeros((n, 2), dtype=np.float32)
    indices = np.arange(n, dtype=np.uint32)
    return verts, indices, normals, uvs


def _make_checkerboard(size: int = 64) -> np.ndarray:
    tex = np.ones((size, size, 4), dtype=np.uint8) * 255
    checker = (np.arange(size)[:, None] // 8 + np.arange(size)[None, :] // 8) % 2 == 0
    tex[checker] = [200, 50, 50, 255]
    tex[~checker] = [50, 50, 200, 255]
    return tex


def _render_and_save(camera, shader, scene, fbo, texture, filename, bg_color=(0.1, 0.1, 0.15, 1.0)):
    fbo.bind()
    texture.bind(0)
    fbo.clear(*bg_color)
    scene.render(camera, shader, extra_uniforms={'texture0': 0})
    pixels = fbo.read(components=4, dtype='u1')
    if not pixels:
        print(f"    ERROR: empty read from FBO")
        return
    img = np.frombuffer(pixels, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    img = np.flipud(img)
    path = os.path.join(OUTPUT_DIR, filename)
    Image.fromarray(img, 'RGBA').save(path)
    print(f"    -> {filename}")


def demo():
    print("=== Standalone Renderer Demo ===\n")

    print("[1] Creating standalone ModernGL context...")
    rctx = RendererContext()
    rctx.create_standalone(WIDTH, HEIGHT)
    ctx = rctx.ctx
    print(f"    Context: {ctx}")

    camera = Camera(aspect=WIDTH / HEIGHT)
    camera.set_position(3.5, 2.5, 4.0)
    camera.set_target(0, 0, 0)

    shader_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'shaders')
    shader = ShaderProgram(
        ctx,
        os.path.join(shader_dir, 'simple.vert'),
        os.path.join(shader_dir, 'simple.frag'),
    )
    print(f"    Shader: {shader.program}")

    tex_data = _make_checkerboard()
    texture = Texture(ctx)
    texture.load_numpy(tex_data)
    print(f"    Texture: {texture.texture}")

    fbo = Framebuffer(ctx, WIDTH, HEIGHT)
    print(f"    FBO: {fbo.fbo}")

    print("\n[2] Rendering passes:\n")

    # --- TRIANGLES ---
    print("  TRIANGLES  (cube)")
    verts, idxs, norms, uvs = _make_cube()
    model_tri = Model(ctx, moderngl.TRIANGLES)
    model_tri.load_from_mesh(verts, idxs, norms, uvs)
    scene1 = Scene(ctx)
    scene1.add_model('cube', model_tri)
    scene1.set_directional_light((0.5, -0.8, -0.3), (1, 1, 1), 1.0)
    _render_and_save(camera, shader, scene1, fbo, texture, 'output_triangles.png')

    # --- TRIANGLE_STRIP ---
    print("  TRIANGLE_STRIP  (quad strip)")
    verts, idxs, norms, uvs = _make_strip()
    model_strip = Model(ctx, moderngl.TRIANGLE_STRIP)
    model_strip.load_from_mesh(verts, idxs, norms, uvs)
    scene2 = Scene(ctx)
    scene2.add_model('strip', model_strip)
    scene2.set_directional_light((0.0, -1.0, 0.0), (1, 1, 1), 1.0)
    cam2 = Camera(aspect=WIDTH / HEIGHT)
    cam2.set_position(2.5, 2.0, 2.5)
    cam2.set_target(0, 0, 0)
    _render_and_save(cam2, shader, scene2, fbo, texture, 'output_strip.png')

    # --- LINES ---
    print("  LINES  (wireframe cube)")
    verts, idxs, norms, uvs = _make_lines()
    model_lines = Model(ctx, moderngl.LINES)
    model_lines.load_from_mesh(verts, idxs, norms, uvs)
    scene3 = Scene(ctx)
    scene3.add_model('lines', model_lines)
    scene3.set_directional_light((0.5, -0.8, -0.3), (1, 1, 1), 1.0)
    cam3 = Camera(aspect=WIDTH / HEIGHT)
    cam3.set_position(3.5, 2.5, 4.0)
    cam3.set_target(0, 0, 0)
    _render_and_save(cam3, shader, scene3, fbo, texture, 'output_lines.png')

    # --- POINTS ---
    print("  POINTS  (random sphere)")
    ctx.point_size = 3.0
    verts, idxs, norms, uvs = _make_points()
    model_pts = Model(ctx, moderngl.POINTS)
    model_pts.load_from_mesh(verts, idxs, norms, uvs)
    scene4 = Scene(ctx)
    scene4.add_model('points', model_pts)
    scene4.set_directional_light((0.5, -0.8, -0.3), (1, 1, 1), 1.0)
    cam4 = Camera(aspect=WIDTH / HEIGHT)
    cam4.set_position(3.0, 1.5, 3.0)
    cam4.set_target(0, 0, 0)
    _render_and_save(cam4, shader, scene4, fbo, texture, 'output_points.png')

    # --- cleanup ---
    model_tri.release()
    model_strip.release()
    model_lines.release()
    model_pts.release()
    texture.release()
    fbo.release()
    ctx.release()

    print(f"\nDone. All outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    demo()
