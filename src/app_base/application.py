import tkinter as tk
from .ui import AppUI
from .controllers import AppController

class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Landscape Editor")
        self.root.geometry("1050x700")
        self.root.minsize(800, 500)

        self.controller = AppController(self.root)
        self.ui = AppUI(self.root, self.controller)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.root.quit()
        self.root.destroy()