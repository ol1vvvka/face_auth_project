import cv2
import pickle
import os
import tkinter as tk
from tkinter import messagebox

# Создаем root для отображения messagebox, но не показываем окно
root = tk.Tk()
root.withdraw()
root.geometry("200x350")
root.resizable(False, False)

model_path = "models/face_model.yml"
labels_path = "models/labels.pkl"

if not os.path.exists(model_path) or not os.path.exists(labels_path):
    messagebox.showerror("Ошибка", "Модель или файл меток не найдены. Сначала обучите модель.")
    exit(1)

try:
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(model_path)
except Exception as e:
    messagebox.showerror("Ошибка загрузки модели", str(e))
    exit(1)

try:
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
except Exception as e:
    messagebox.showerror("Ошибка загрузки меток", str(e))
    exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    messagebox.showerror("Ошибка", "Камера не доступна.")
    exit(1)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Устанавливаем название окна для распознавания на английском
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

messagebox.showinfo("Распознавание", "Распознавание запущено. Нажмите ESC для выхода.")

while True:
     # считывание кадра с камеры
    ret, frame = cap.read()
    # если не удалось - предупреждаем пользователя
    if not ret:
        messagebox.showwarning("Предупреждение", "Не удалось считать кадр.")
        continue

    # преобразование изображения в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # находим лица на изображении
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # получение области лица
        face = gray[y:y+h, x:x+w]
        # предсказание метки (id) и уровеня уверенности - 30-60 в пределах нормы, выше - не найден
        label, confidence = model.predict(face)
        # получение имени по метке
        name = labels.get(label, "Unknown")

        # формирование текста с именем и уверенностью
        text = f"{name} ({confidence:.1f})"
        # отображение имени и рамки вокруг лица
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Показываем изображение в окне с новым названием
    cv2.imshow("Face Recognition", frame)

    # прерывание цикла по нажатию клавиши ESC
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
