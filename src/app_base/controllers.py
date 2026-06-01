import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class AppController:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_image: Image.Image | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.view_mode = tk.StringVar(value="2D")
        self.image_label: tk.Label | None = None

    def set_image_label(self, label: tk.Label):
        self.image_label = label

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if not file_path:
            return

        try:
            self.current_image = Image.open(file_path)
            self._display_image()
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", f"Не удалось открыть файл:\n{e}")

    def save_image(self):
        if not self.current_image:
            messagebox.showwarning("Предупреждение", "Нет изображения для сохранения.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if file_path:
            self.current_image.save(file_path)
            messagebox.showinfo("Успех", "Изображение сохранено.")

    def toggle_view(self, mode: str):
        self.view_mode.set(mode)
        print(f"[Controller] Режим вида изменён на: {mode}")

    def generate_landscape(self):
        messagebox.showinfo("Генерация", "Запуск генерации ландшафта... (заглушка)")

    def _display_image(self):
        if not self.current_image or not self.image_label:
            return

        img_copy = self.current_image.copy()
        img_copy.thumbnail((800, 600), Image.Resampling.LANCZOS)
        
        self.photo_image = ImageTk.PhotoImage(img_copy)
        self.image_label.config(image=self.photo_image, text="")