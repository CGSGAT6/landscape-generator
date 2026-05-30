import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .controllers import AppController

class AppUI:
    def __init__(self, root: tk.Tk, controller: "AppController"):
        self.controller = controller
        self._build_toolbar(root)
        self._build_workspace(root)
        self._build_settings_panel(root)

    def _build_toolbar(self, parent: tk.Tk):
        toolbar = ttk.Frame(parent)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(toolbar, text="Load", command=self.controller.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.controller.save_image).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        ttk.Label(toolbar, text="Режим:").pack(side=tk.LEFT, padx=(0, 4))
        view_cb = ttk.Combobox(
            toolbar,
            textvariable=self.controller.view_mode,
            values=["2D", "3D"],
            state="readonly",
            width=5
        )
        view_cb.pack(side=tk.LEFT)
        view_cb.bind("<<ComboboxSelected>>", lambda e: self.controller.toggle_view(e.widget.get()))

    def _build_workspace(self, parent: tk.Tk):
        main_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.image_label = ttk.Label(
            main_frame,
            background="#f0f0f0",
            text="Нажмите Load, чтобы загрузить изображение",
            anchor=tk.CENTER
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.controller.set_image_label(self.image_label)

    def _build_settings_panel(self, parent: tk.Tk):
        panel = ttk.LabelFrame(parent, text="Параметры ландшафта", padding=10)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        ttk.Label(panel, text="Высота (м):").pack(anchor=tk.W, pady=(0, 2))
        ttk.Scale(panel, from_=0, to=100, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        ttk.Label(panel, text="Шум Перлина:").pack(anchor=tk.W, pady=(4, 2))
        ttk.Scale(panel, from_=0.0, to=1.0, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        ttk.Label(panel, text="Детализация:").pack(anchor=tk.W, pady=(4, 2))
        ttk.Spinbox(panel, from_=1, to=16, width=6).pack(anchor=tk.W, pady=2)

        ttk.Button(panel, text="Generate", command=self.controller.generate_landscape).pack(pady=(12, 4), fill=tk.X)