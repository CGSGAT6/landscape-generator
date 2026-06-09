from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .controllers import AppController


class AppUI:
    def __init__(self, root: tk.Tk, controller: AppController):
        self.controller = controller
        self.controller._ctrl_pressed = False
        self.controller._shift_pressed = False
        self._build_toolbar(root)
        self._build_settings_panel(root)
        self._build_workspace(root)
        self._bind_keys(root)

    def _build_toolbar(self, parent: tk.Tk) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(toolbar, text="Load", command=self.controller.load_landscape).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.controller.save_landscape).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        ttk.Label(toolbar, text="Mode:").pack(side=tk.LEFT, padx=(0, 4))
        view_cb = ttk.Combobox(
            toolbar,
            textvariable=self.controller.view_mode,
            values=["2D", "3D"],
            state="readonly",
            width=5,
        )
        view_cb.pack(side=tk.LEFT)
        view_cb.bind("<<ComboboxSelected>>", lambda e: self.controller.toggle_view(e.widget.get()))

        ttk.Checkbutton(toolbar, text="Flat", variable=self.controller.show_flat,
                        command=self.controller.toggle_flat).pack(side=tk.LEFT, padx=4)

    def _build_settings_panel(self, parent: tk.Tk) -> None:
        panel = ttk.LabelFrame(parent, text="Landscape parameters", padding=10)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        ttk.Label(panel, text="Main parameters").pack(anchor=tk.W, pady=(0, 4))
        self._add_slider(panel, "Detail:", self.controller.detail, 0, 100)
        self._add_slider(panel, "Noise decay:", self.controller.noise_decay, 0.0, 1.0)
        self._add_slider(panel, "Mountain height:", self.controller.mountain_height, 0, 100)
        self._add_slider(panel, "Frequency:", self.controller.frequency, 0.01, 0.5)
        self._add_slider(panel, "Lacunarity:", self.controller.lacunarity, 1.0, 4.0)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(panel, text="Seed and size").pack(anchor=tk.W, pady=(0, 4))
        self._add_entry(panel, "Seed 1:", self.controller.seed1, width=12)
        self._add_entry(panel, "Seed 2:", self.controller.seed2, width=12)
        self._add_entry(panel, "Width:", self.controller.width, width=12, is_int=True)
        self._add_entry(panel, "Height:", self.controller.height, width=12, is_int=True)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(panel, text="Visualisation").pack(anchor=tk.W, pady=(0, 4))
        self._add_slider(panel, "Tree density:", self.controller.tree_density, 0, 100)
        self._add_slider(panel, "Power:", self.controller.power, 0.3, 3.0)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        generate_btn = ttk.Button(panel, text="GENERATE", command=self.controller.generate_landscape)
        generate_btn.pack(fill=tk.X, pady=(4, 8), ipady=8)

    def _add_slider(self, parent, label_text: str, variable: tk.Variable, from_: float, to: float) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        ttk.Label(frame, text=label_text, width=18).pack(side=tk.LEFT)

        ttk.Scale(
            frame,
            variable=variable,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        ttk.Label(frame, textvariable=variable, width=6, anchor=tk.E).pack(side=tk.LEFT, padx=(4, 0))

    def _add_entry(self, parent, label_text: str, variable: tk.Variable, width: int = 10, is_int: bool = False) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        ttk.Label(frame, text=label_text, width=18).pack(side=tk.LEFT)

        entry = ttk.Entry(frame, textvariable=variable, width=width)
        entry.pack(side=tk.LEFT, padx=(4, 0))

        if is_int:
            def validate_int(value):
                if value == "" or value.lstrip("-").isdigit():
                    return True
                return False
            vcmd = (entry.register(validate_int), "%P")
            entry.configure(validate="key", validatecommand=vcmd)

    def _build_workspace(self, parent: tk.Tk) -> None:
        main_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.image_label = ttk.Label(
            main_frame,
            background="#f0f0f0",
            text="Press Load or GENERATE",
            anchor=tk.CENTER,
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.controller.set_image_label(self.image_label)

        self.image_label.bind("<ButtonPress-1>", self._on_mouse_down)
        self.image_label.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.image_label.bind("<B1-Motion>", self._on_mouse_drag)
        self.image_label.bind("<ButtonPress-3>", self._on_mouse_down)
        self.image_label.bind("<ButtonRelease-3>", self._on_mouse_up)
        self.image_label.bind("<B3-Motion>", self._on_mouse_drag)
        self.image_label.bind("<MouseWheel>", self._on_mouse_scroll)
        self.image_label.bind("<Button-4>", self._on_mouse_scroll)
        self.image_label.bind("<Button-5>", self._on_mouse_scroll)

    def _bind_keys(self, root: tk.Tk) -> None:
        c = self.controller
        root.bind("<Left>", lambda e: c._held_keys.add("left"))
        root.bind("<KeyRelease-Left>", lambda e: c._held_keys.discard("left"))
        root.bind("<Right>", lambda e: c._held_keys.add("right"))
        root.bind("<KeyRelease-Right>", lambda e: c._held_keys.discard("right"))
        root.bind("<Up>", lambda e: c._held_keys.add("up"))
        root.bind("<KeyRelease-Up>", lambda e: c._held_keys.discard("up"))
        root.bind("<Down>", lambda e: c._held_keys.add("down"))
        root.bind("<KeyRelease-Down>", lambda e: c._held_keys.discard("down"))
        root.bind("<equal>", lambda e: c._held_keys.add("zoom_in"))
        root.bind("<KeyRelease-equal>", lambda e: c._held_keys.discard("zoom_in"))
        root.bind("<minus>", lambda e: c._held_keys.add("zoom_out"))
        root.bind("<KeyRelease-minus>", lambda e: c._held_keys.discard("zoom_out"))
        root.bind("<r>", lambda e: c.reload_shaders())
        root.bind("<R>", lambda e: c.reload_shaders())
        root.bind("<a>", lambda e: c.toggle_auto_orbit())
        root.bind("<A>", lambda e: c.toggle_auto_orbit())
        root.bind("<c>", lambda e: c.camera_home())
        root.bind("<C>", lambda e: c.camera_home())
        root.bind("<f>", lambda e: c.toggle_fullscreen())
        root.bind("<F>", lambda e: c.toggle_fullscreen())
        root.bind("<Control_L>", lambda e: setattr(c, "_ctrl_pressed", True))
        root.bind("<Control_R>", lambda e: setattr(c, "_ctrl_pressed", True))
        root.bind("<KeyRelease-Control_L>", lambda e: setattr(c, "_ctrl_pressed", False))
        root.bind("<KeyRelease-Control_R>", lambda e: setattr(c, "_ctrl_pressed", False))
        root.bind("<Shift_L>", lambda e: setattr(c, "_shift_pressed", True))
        root.bind("<Shift_R>", lambda e: setattr(c, "_shift_pressed", True))
        root.bind("<KeyRelease-Shift_L>", lambda e: setattr(c, "_shift_pressed", False))
        root.bind("<KeyRelease-Shift_R>", lambda e: setattr(c, "_shift_pressed", False))

    def _on_mouse_down(self, event: tk.Event) -> None:
        self.controller._mouse_pressed = True
        self.controller._mouse_last_x = event.x
        self.controller._mouse_last_y = event.y

    def _on_mouse_up(self, event: tk.Event) -> None:
        self.controller._mouse_pressed = False

    def _on_mouse_drag(self, event: tk.Event) -> None:
        ctrl = self.controller
        dx = event.x - ctrl._mouse_last_x
        dy = event.y - ctrl._mouse_last_y
        ctrl._mouse_last_x = event.x
        ctrl._mouse_last_y = event.y

        mult = 3.0 if ctrl._shift_pressed else 1.0

        if ctrl._ctrl_pressed:
            ctrl.camera_pan(-dx * 0.01 * mult, dy * 0.01 * mult)
        else:
            ctrl.camera_orbit(-dx * 0.25 * mult, dy * 0.25 * mult)

    def _on_mouse_scroll(self, event: tk.Event) -> None:
        ctrl = self.controller
        delta = event.delta if hasattr(event, "delta") else (
            1 if event.num == 4 else -1
        )
        mult = 3.0 if ctrl._shift_pressed else 1.0
        ctrl.camera_zoom(delta * mult)
