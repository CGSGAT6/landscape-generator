from __future__ import annotations

import time
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from .render_bridge import RenderBridge
from landscape import Landscape
from landscape.generator import Generator


class AppController:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_image: Image.Image | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.view_mode = tk.StringVar(value="2D")
        self.image_label: tk.Label | None = None

        self._generator = Generator()
        self._landscape: Landscape | None = None
        self._from_file: bool = False
        self._bridge: RenderBridge | None = None
        self._loop_job: str | None = None
        self._last_time: float = 0.0
        self._auto_orbit: bool = False
        self._orbit_speed: float = 0.5

        self._mouse_pressed: bool = False
        self._mouse_last_x: int = 0
        self._mouse_last_y: int = 0
        self._ctrl_pressed: bool = False
        self._shift_pressed: bool = False
        self._held_keys: set = set()

        self.detail = tk.DoubleVar(value=50.0)
        self.noise_decay = tk.DoubleVar(value=0.5)
        self.mountain_height = tk.DoubleVar(value=50.0)
        self.frequency = tk.DoubleVar(value=0.05)
        self.lacunarity = tk.DoubleVar(value=2.0)

        self.seed1 = tk.StringVar(value="12345")
        self.seed2 = tk.StringVar(value="67890")
        self.width = tk.IntVar(value=256)
        self.height = tk.IntVar(value=256)

        self.tree_density = tk.DoubleVar(value=30.0)
        self.power = tk.DoubleVar(value=1.0)
        self.show_flat = tk.BooleanVar(value=False)

        root.bind("<Configure>", self._on_resize, add="+")

    def set_image_label(self, label: tk.Label):
        self.image_label = label

    def load_landscape(self):
        directory = filedialog.askdirectory(title="Load landscape")
        if not directory:
            return
        try:
            landscape = Landscape.load(directory)
            self._landscape = landscape
            self._from_file = True
            self._display_heightmap(landscape)
            if self._bridge is not None:
                self._bridge.set_landscape(landscape, from_file=True)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load landscape:\n{e}")

    def save_landscape(self):
        if self._landscape is None:
            messagebox.showwarning("Warning", "No landscape to save.")
            return
        directory = filedialog.askdirectory(title="Save landscape")
        if not directory:
            return
        try:
            self._landscape.save(directory)
            messagebox.showinfo("Success", "Landscape saved.")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save landscape:\n{e}")

    # ── 2D / 3D ──────────────────────────────────────────────────────

    def toggle_view(self, mode: str) -> None:
        self.view_mode.set(mode)
        if mode == "3D":
            self._start_3d()
        else:
            self._stop_3d()

    def _start_3d(self) -> None:
        if self._bridge is not None:
            return
        w = self.image_label.winfo_width()
        h = self.image_label.winfo_height()
        if w < 2 or h < 2:
            w, h = 800, 600
        self._bridge = RenderBridge(w, h)
        self._bridge.resize(w, h)
        if self._landscape is not None:
            self._bridge.set_landscape(self._landscape, from_file=self._from_file)
        self._last_time = time.perf_counter()
        self._auto_orbit = False
        self._render_loop()

    def _stop_3d(self) -> None:
        if self._loop_job is not None:
            self.root.after_cancel(self._loop_job)
            self._loop_job = None
        self._held_keys.clear()
        if self._bridge is not None:
            self._bridge.release()
            self._bridge = None
        self._display_image()

    def _render_loop(self) -> None:
        if self._bridge is None:
            return
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        mult = 3.0 if self._shift_pressed else 1.0
        orbit_speed = 60.0 * mult
        pan_speed = 2.0 * mult
        zoom_speed = 3.0 * mult

        if "up" in self._held_keys:
            self._auto_orbit = False
            if self._ctrl_pressed:
                self._bridge.renderer.camera_pan(0, pan_speed * dt)
            else:
                self._bridge.renderer.camera_orbit(0, orbit_speed * dt)
        if "down" in self._held_keys:
            self._auto_orbit = False
            if self._ctrl_pressed:
                self._bridge.renderer.camera_pan(0, -pan_speed * dt)
            else:
                self._bridge.renderer.camera_orbit(0, -orbit_speed * dt)
        if "left" in self._held_keys:
            self._auto_orbit = False
            if self._ctrl_pressed:
                self._bridge.renderer.camera_pan(-pan_speed * dt, 0)
            else:
                self._bridge.renderer.camera_orbit(-orbit_speed * dt, 0)
        if "right" in self._held_keys:
            self._auto_orbit = False
            if self._ctrl_pressed:
                self._bridge.renderer.camera_pan(pan_speed * dt, 0)
            else:
                self._bridge.renderer.camera_orbit(orbit_speed * dt, 0)
        if "zoom_in" in self._held_keys:
            self._auto_orbit = False
            self._bridge.renderer.camera_zoom(zoom_speed * dt)
        if "zoom_out" in self._held_keys:
            self._auto_orbit = False
            self._bridge.renderer.camera_zoom(-zoom_speed * dt)

        if self._auto_orbit:
            self._bridge.renderer.camera_orbit(self._orbit_speed * 30 * dt, 0.0)

        img = self._bridge.render(now, dt)
        self.photo_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo_image, text="")

        self._loop_job = self.root.after(16, self._render_loop)

    def _on_resize(self, event: tk.Event | None = None) -> None:
        if self._bridge is None or self.image_label is None:
            return
        w = self.image_label.winfo_width()
        h = self.image_label.winfo_height()
        if w > 1 and h > 1:
            self._bridge.resize(w, h)

    # ── Camera proxy ─────────────────────────────────────────────────

    def camera_orbit(self, dx: float, dy: float) -> None:
        if self._bridge is not None:
            self._auto_orbit = False
            self._bridge.renderer.camera_orbit(dx, dy)

    def camera_zoom(self, delta: float) -> None:
        if self._bridge is not None:
            self._auto_orbit = False
            self._bridge.renderer.camera_zoom(delta)

    def camera_pan(self, dx: float, dy: float) -> None:
        if self._bridge is not None:
            self._auto_orbit = False
            self._bridge.renderer.camera_pan(dx, dy)

    def camera_home(self) -> None:
        if self._bridge is not None:
            self._auto_orbit = False
            from renderer.camera import Camera
            cam = Camera()
            self._bridge.renderer.camera = cam

    def toggle_auto_orbit(self) -> None:
        if self._bridge is not None:
            self._auto_orbit = not self._auto_orbit

    def reload_shaders(self) -> None:
        if self._bridge is not None:
            self._bridge.renderer.reload_shaders()
            print("Shaders reloaded")

    def toggle_fullscreen(self) -> None:
        current = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not current)

    def toggle_flat(self) -> None:
        if self._bridge is not None:
            self._bridge.set_view_flat(self.show_flat.get())

    # ── Generation ───────────────────────────────────────────────────

    def generate_landscape(self) -> dict | None:
        params = {
            "detail": self.detail.get(),
            "noise_decay": self.noise_decay.get(),
            "mountain_height": self.mountain_height.get(),
            "frequency": self.frequency.get(),
            "seed1": self.seed1.get(),
            "seed2": self.seed2.get(),
            "width": self.width.get(),
            "height": self.height.get(),
            "tree_density": self.tree_density.get(),
            "power": self.power.get(),
        }

        print("\n" + "=" * 40)
        print("LANDSCAPE GENERATION")
        print("=" * 40)
        for key, value in params.items():
            print(f"  {key:15} → {value}")
        print("=" * 40 + "\n")

        def _hash_seed(s: str) -> int:
            try:
                return int(s) & 0x7FFFFFFF
            except ValueError:
                return hash(s) & 0x7FFFFFFF

        landscape = self._generator.generate(
            width=self.width.get(),
            height=self.height.get(),
            seed_height=_hash_seed(self.seed1.get()),
            seed_moisture=_hash_seed(self.seed2.get()),
            z_scale=self.mountain_height.get(),
            detail=self.detail.get(),
            noise_decay=self.noise_decay.get(),
            frequency=self.frequency.get(),
            lacunarity=self.lacunarity.get(),
            power=self.power.get(),
            tree_density=self.tree_density.get(),
        )
        self._landscape = landscape

        self._display_heightmap(landscape)

        if self._bridge is not None:
            self._bridge.set_landscape(landscape)

        print(f"Generated landscape {landscape.height_map.shape}")
        return params

    def _display_heightmap(self, landscape: Landscape) -> None:
        hm = landscape.height_map
        hm_norm = ((hm - hm.min()) / (hm.max() - hm.min() + 1e-8) * 255).astype(np.uint8)
        w = max(self.image_label.winfo_width(), 100)
        h = max(self.image_label.winfo_height(), 100)
        img = Image.fromarray(hm_norm, mode="L").resize((w, h), Image.Resampling.NEAREST)
        self.current_image = img
        self.photo_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo_image, text="")

    # ── Image display ────────────────────────────────────────────────

    def _display_image(self) -> None:
        if not self.current_image or not self.image_label:
            return
        img_copy = self.current_image.copy()
        img_copy.thumbnail((800, 600), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(img_copy)
        self.image_label.config(image=self.photo_image, text="")

    # ── Cleanup ──────────────────────────────────────────────────────

    def release(self) -> None:
        self._stop_3d()
