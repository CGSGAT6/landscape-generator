import moderngl_window as mglw
from renderer.renderer import RenderEngine
import numpy as np
import pyrr
from filepath import *
from landscape import Landscape


class TestWindow(mglw.WindowConfig):
    gl_version = (4, 3)
    window_size = (512, 512)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.renderer = RenderEngine()
        self.renderer.create_from_window(self.ctx)
        self.renderer.init()
        self.renderer.set_screen_fbo(self.ctx.screen)
        self.renderer.resize(*self.wnd.size)

        self._mouse_pressed = False
        self._ctrl_pressed = False
        self._shift_pressed = False
        self._auto_orbit = False
        self._held_keys = set()
        self._speed_mult = 3.0

        self.renderer.camera.set_position(1, 7, 10)
        cam = self.renderer.camera
        self._home = (
            cam._target.copy(),
            cam._distance,
            cam._yaw,
            cam._pitch,
        )
        s = 5
        v = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [s, 0, 0, 0, 1, 0, 1, 0],
            [0, s, 0, 0, 1, 0, 0, 1],
        ], dtype=np.float32)

        self.renderer.set_clear_color((0.1 / 2, 0.1 / 2, 0.1 / 2, 0.1 / 2))

        self.tr = self.renderer.create_prim(vertices=v,
                                            mtl=self.renderer.get_material("tex test",
                                                                           tex=[OUTPUT_DIR / "heightmap.png"]))
        #self.mdl = self.renderer.create_model(name="test model", prims=[])
        #self.mdl.add_from_file(fname=OUTPUT_DIR / "OldCar/oldcar.obj", flat=False)
        self.land = Landscape.load(OUTPUT_DIR/"demo_landscape")
        #self.land_pr = self.renderer.create_prim(name="land_pr", vertices=self.land.flat_mesh.to_interleaved(),
        #                                         indeces=self.land.grid_mesh.indices)
        #self.mdl = self.renderer.create_model(name="test model", prims=[self.land_pr])
        self.mdl = self.renderer.create_model(name="test model", file_path=self.land.grid_mesh_path)


    def on_render(self, time, frame_time):
        self.renderer.begin_frame(time, frame_time)

        if self._auto_orbit:
            self.renderer.camera_orbit(30 * frame_time, 0)

        mult = self._speed_mult if self._shift_pressed else 1.0
        orbit_speed = 60.0 * mult
        pan_speed = 2.0 * mult
        zoom_speed = 3.0 * mult

        if self.wnd.keys.UP in self._held_keys:
            self._auto_orbit = False
            if self._ctrl_pressed:
                self.renderer.camera_pan(0, pan_speed * frame_time)
            else:
                self.renderer.camera_orbit(0, orbit_speed * frame_time)
        if self.wnd.keys.DOWN in self._held_keys:
            self._auto_orbit = False
            if self._ctrl_pressed:
                self.renderer.camera_pan(0, -pan_speed * frame_time)
            else:
                self.renderer.camera_orbit(0, -orbit_speed * frame_time)
        if self.wnd.keys.LEFT in self._held_keys:
            self._auto_orbit = False
            if self._ctrl_pressed:
                self.renderer.camera_pan(-pan_speed * frame_time, 0)
            else:
                self.renderer.camera_orbit(-orbit_speed * frame_time, 0)
        if self.wnd.keys.RIGHT in self._held_keys:
            self._auto_orbit = False
            if self._ctrl_pressed:
                self.renderer.camera_pan(pan_speed * frame_time, 0)
            else:
                self.renderer.camera_orbit(orbit_speed * frame_time, 0)
        if self.wnd.keys.EQUAL in self._held_keys:
            self._auto_orbit = False
            self.renderer.camera_zoom(zoom_speed * frame_time)
        if self.wnd.keys.MINUS in self._held_keys:
            self._auto_orbit = False
            self.renderer.camera_zoom(-zoom_speed * frame_time)

        bb = (self.mdl.bbox_max + self.mdl.bbox_min) / 2
        self.mdl.render(pyrr.Matrix44.from_translation(-bb) * pyrr.Matrix44.from_scale((0.5, 0.5, 0.5)))
        self.renderer.end_frame()

    def on_resize(self, width: int, height: int):
        self.renderer.resize(*self.wnd.size)

    def on_close(self):
        pass

    # --- Keyboard ---

    def on_key_event(self, key, action, modifiers) -> None:
        self._ctrl_pressed = bool(modifiers.ctrl)
        self._shift_pressed = bool(modifiers.shift)

        if action == self.wnd.keys.ACTION_PRESS:
            self._held_keys.add(key)
        elif action == self.wnd.keys.ACTION_RELEASE:
            self._held_keys.discard(key)
            return
        else:
            return

        if key == self.wnd.keys.R:
            self.renderer.reload_shaders()
        elif key == self.wnd.keys.A:
            self._auto_orbit = not self._auto_orbit
        elif key == self.wnd.keys.F:
            self.wnd.fullscreen = not self.wnd.fullscreen
        elif key == self.wnd.keys.C:
            cam = self.renderer.camera
            target, dist, yaw, pitch = self._home
            cam._target[:] = target
            cam._distance = dist
            cam._yaw = yaw
            cam._pitch = pitch
            cam.recalculate()
            self.renderer._update_camera_buf()
            self._auto_orbit = False

    # --- Mouse ---

    def on_mouse_press_event(self, x: int, y: int, button: int):
        if button == 1:
            self._mouse_pressed = True

    def on_mouse_release_event(self, x: int, y: int, button: int):
        if button == 1:
            self._mouse_pressed = False

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        if not self._mouse_pressed:
            return
        self._auto_orbit = False
        mult = self._speed_mult if self._shift_pressed else 1.0
        if self._ctrl_pressed:
            self.renderer.camera_pan(-dx * 0.01 * mult, dy * 0.01 * mult)
        else:
            self.renderer.camera_orbit(-dx * 0.25 * mult, dy * 0.25 * mult)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        self._auto_orbit = False
        mult = self._speed_mult if self._shift_pressed else 1.0
        self.renderer.camera_zoom(y_offset * mult)


def demo():
    mglw.run_window_config(TestWindow)


if __name__ == "__main__":
    demo()
