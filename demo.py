import sys
import os

# Добавляем корень проекта в путь поиска модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app_base.application import Application

def main():
    print("🚀 Запуск Landscape Application...")
    app = Application()
    app.run()
    print("✅ Приложение завершено.")

if __name__ == "__main__":
    main()