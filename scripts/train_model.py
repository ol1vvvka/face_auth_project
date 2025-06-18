import cv2
import os
import numpy as np
import pickle
import tkinter as tk
from tkinter import messagebox



# Инициализация Tkinter без отображения окна
root = tk.Tk()
root.withdraw()

dataset_path = "dataset"
models_path = "models"

if not os.path.exists(dataset_path):
    messagebox.showerror("Ошибка", f"Папка с датасетом '{dataset_path}' не найдена.")
    exit(1)

faces = []
labels = []
label_map = {}
label_id = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                messagebox.showwarning("Предупреждение", f"Пропущено (не удалось загрузить): {img_path}")
                continue
            faces.append(img)
            labels.append(label_id)
        except Exception as e:
            messagebox.showwarning("Ошибка", f"Ошибка при загрузке изображения {img_path}:\n{e}")

    label_id += 1

if not faces:
    messagebox.showerror("Ошибка", "Не найдено ни одного корректного изображения для обучения.")
    exit(1)

try:
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))
except Exception as e:
    messagebox.showerror("Ошибка обучения", str(e))
    exit(1)

try:
    os.makedirs(models_path, exist_ok=True)
    model.save(os.path.join(models_path, "face_model.yml"))

    with open(os.path.join(models_path, "labels.pkl"), "wb") as f:
        pickle.dump(label_map, f)

    messagebox.showinfo("Готово", "Обучение завершено успешно.")
except Exception as e:
    messagebox.showerror("Ошибка сохранения", str(e))
