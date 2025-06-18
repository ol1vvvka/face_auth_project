import cv2
import os
import sys
import tkinter as tk
from tkinter import messagebox

app = tk.Tk()
app.geometry("200x350")

# Инициализация tkinter (без главного окна)
root = tk.Tk()
root.withdraw()

# Получаем имя пользователя из аргументов командной строки
if len(sys.argv) < 2:
    messagebox.showerror("Ошибка", "Имя пользователя не указано.")
    sys.exit(1)

user_name = sys.argv[1]
save_path = f"dataset/{user_name}"

# Создаем директорию для сохранения изображений, если её нет
os.makedirs(save_path, exist_ok=True)

# Открываем камеру
cap = cv2.VideoCapture(0)

# Проверяем, что камера открыта
if not cap.isOpened():
    messagebox.showerror("Ошибка", "Не удалось открыть камеру.")
    sys.exit(1)

# Инициализация каскадного классификатора для распознавания лиц
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
while count < 50:
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Ошибка", "Не удалось считать кадр с камеры.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        try:
            cv2.imwrite(f"{save_path}/{count}.jpg", face)
        except Exception as e:
            messagebox.showwarning("Ошибка сохранения", f"Ошибка при сохранении изображения {count}.jpg:\n{e}")
            continue
        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Сбор лиц", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

if count >= 50:
    messagebox.showinfo("Готово", f"Сбор изображений завершён. Сохранено: {count}")
else:
    messagebox.showwarning("Внимание", f"Сбор завершён досрочно. Сохранено: {count} изображений.")
    
# Название окна
cv2.imshow("Image Collection", frame)
