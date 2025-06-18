import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import sys
import cv2 
from PIL import Image, ImageTk

app = tk.Tk()

camera_active = False
cap = None

app.title("FaceAuth GUI")
app.geometry("600x650")
app.resizable(False, False)

# Получаем базовую директорию (на уровень выше текущего файла)
try:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    base_dir = os.path.abspath("..")

# Установка иконки
icon_path = os.path.join(base_dir, "icon.ico")
if os.path.exists(icon_path):
    try:
        app.iconbitmap(icon_path)
    except Exception as e:
        print(f"Ошибка установки иконки: {e}")

# Заголовок
tk.Label(app, text="Добро пожаловать в FaceAuth", font=("Arial", 14)).pack(pady=10)

# Ввод имени
tk.Label(app, text="Имя нового пользователя:", font=("Arial", 10)).pack()
username_entry = tk.Entry(app, width=30)
username_entry.pack(pady=5)

# Универсальная функция запуска скриптов
def run_script(script_name, with_arg=False):
    script_path = os.path.join(base_dir, "scripts", script_name)
    if not os.path.exists(script_path):
        messagebox.showerror("Ошибка", f"Скрипт не найден: {script_path}")
        return

    command = [sys.executable, script_path]
    if with_arg:
        username = username_entry.get().strip()
        if not username:
            messagebox.showwarning("Внимание", "Введите имя пользователя")
            return
        command.append(username)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Ошибка", f"Скрипт завершился с ошибкой:\n{e}")
    except Exception as e:
        messagebox.showerror("Ошибка запуска", str(e))

# Показываем камеру внутри приложения
cap = None  # глобальный объект для камеры

def show_camera():
    global cap, camera_active
    if camera_active:
        return  # Камера уже запущена

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Ошибка", "Не удалось подключиться к камере.")
        return

    camera_active = True

    def update_frame():
        if not camera_active:
            return

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image_tk = ImageTk.PhotoImage(image)

        canvas.create_image(0, 0, image=image_tk, anchor=tk.NW)
        canvas.image = image_tk
        canvas.after(30, update_frame)

    update_frame()

def stop_camera():
    global cap, camera_active
    if cap is not None and cap.isOpened():
        cap.release()
    camera_active = False
    canvas.delete("all")  # очистить canvas

# Canvas для вывода камеры
canvas = tk.Canvas(app, width=550, height=400, bg="black")
canvas.pack(pady=10)

# Первый ряд кнопок
top_buttons_frame = tk.Frame(app)
top_buttons_frame.pack(pady=5)

tk.Button(top_buttons_frame, text="Сбор изображений", width=20, command=lambda: run_script("collect_faces.py", with_arg=True)).pack(side=tk.LEFT, padx=5)
tk.Button(top_buttons_frame, text="Обучение модели", width=20, command=lambda: run_script("train_model.py")).pack(side=tk.LEFT, padx=5)
tk.Button(top_buttons_frame, text="Распознавание лиц", width=20, command=lambda: run_script("face_recognizer.py")).pack(side=tk.LEFT, padx=5)

# Второй ряд кнопок
bottom_buttons_frame = tk.Frame(app)
bottom_buttons_frame.pack(pady=5)

tk.Button(bottom_buttons_frame, text="Проверка камеры", width=20, command=show_camera).pack(side=tk.LEFT, padx=5)
tk.Button(bottom_buttons_frame, text="Остановить камеру", width=20, command=stop_camera).pack(side=tk.LEFT, padx=5)
tk.Button(bottom_buttons_frame, text="Выход", width=15, command=app.quit).pack(side=tk.LEFT, padx=5)



# Закрытие камеры при выходе
def on_close():
    global cap
    if cap is not None:
        cap.release()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)
app.mainloop()
