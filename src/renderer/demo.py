import moderngl_window as mglw
from renderer.renderer import RenderEngine
import numpy as np
import pyrr

class TestWindow(mglw.WindowConfig):
    gl_version = (4, 3)
    window_size=(512, 512)    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.renderer = RenderEngine()
        self.renderer.create_from_window(self.ctx)
        self.renderer.init()
        self.renderer.set_screen_fbo(self.ctx.screen)
        self.renderer.resize(*self.wnd.size)

        self.renderer.camera.set_position(0, 7, -10)
        s = 5
        v = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [s, 0, 0, 0, 1, 0, 1, 0],
            [0, s, 0, 0, 1, 0, 0, 1]
        ], dtype=np.float32)

        self.tr = self.renderer.create_prim(vertices=v)

    def on_render(self, time, frame_time):
        self.renderer.begin_frame(time, frame_time)
        self.renderer.camera.orbit(30 * frame_time, 0)
        self.renderer._update_camera_buf()
        self.renderer.render_prim(self.tr, pyrr.Matrix44.from_x_rotation(2 * 3.14  * time))
        self.renderer.end_frame()

    def on_resize(self, width: int, height: int):
        self.renderer.resize(*self.wnd.size)
        
    def on_close(self):
        pass    
    
def demo():
    mglw.run_window_config(TestWindow)

if __name__ == "__main__":
    demo()