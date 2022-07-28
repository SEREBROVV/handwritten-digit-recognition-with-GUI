import cv2
from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageDraw, ImageGrab, Image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model_mnist.h5')

class App(tk.Tk):
    """Отвечает за построение графического интерфейса
       для изображения цифры пользователем и активации кнопки, которая
       активирует функцию get_answer и отображает результат"""
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Создание элементов
        self.canvas = tk.Canvas(self, width=300, height=300, bd=0, highlightthickness=0, cursor="circle")   # создаём полотно
        self.label = tk.Label(self, text='Нарисуйте цифру', font=("Helvetica", 24))     # надпись на полотне
        self.classify_btn = tk.Button(self, text="Распознать", command=self.classify_handwriting)   # первая кнопка
        self.button_clear = tk.Button(self, text="Очистить", command=self.clear_all)    # вторая кнопка

        # Сетка окна(где расположены элементы: текст, кнопки)
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # если мышь с левой зажатой кнопкой двигается, то вызываем функцию draw_lines
        self.canvas.bind("<B1-Motion>", self.draw)

    def get_answer(self):
        """1-ая часть функции: Получает скрин определённой части экрана,
           в которой нарисованна цифра, сохраняет её в файл image_0.png,
           конвертирует полученное изображение в черно-белое и извлекает
           контуры полученного изображения в переменную counters."""

        self.update()
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width() + 100
        y1 = y + self.canvas.winfo_height() + 100

        image_number = 0
        filename = f'image_{image_number}.png'
        ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

        # читаем изображение в цветном формате
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        # конвертируем фотографию в черно-белую
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # метод threshold() для каждого пикселя применяется одно и то же пороговое значение : если значение пикселя
        # меньше порогового значения оно устанавляется как 0, в противном случаем оно устанавливается на максимальное
        # значение. Первый аргумент - это искодное изображение, которое должно быть в градациях серого. Второй аргумент-
        # это пороговое значение, третий - максимальное значение. Четвертый аргумент - различные типы пороговых значений
        # метод возращает два результата. Первый - это использованный порого и второй - изображение с пороговый значен.
        ret, th = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # findCounter() - функция, которая извлекает контуры фотографии
        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        """2 часть функции: создаёт ограничивающие рамки для контуров и вычисляет
           roi - наилучшее изображение, которое нам подходит. Даёт данное изображение
           полученной нейросети и распознаёт цифру."""
        for cnt in contours:
            # получаем ограничивающую рамку и извлекаем ROI
            x, y, w, h = cv2.boundingRect(cnt)
            # создаём прямоугольник
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            top = int(0.05 * th.shape[0])
            bottom = top
            left = int(0.05 * th.shape[1])
            right = left
            th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
            # извлекаем фотографию ROI
            roi = th[y-top:y+h+bottom, x-left:x+w+right]
            try:
                img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            except:
                continue
            img = img.reshape(1, 28, 28, 1)
            img = img/255.0
            pred = model.predict([img])[0]
            final_pred = np.argmax(pred)
            plt.imshow(roi)
            plt.show()
            break
        return final_pred, pred[final_pred]


    def clear_all(self):
        """Удаляет всё, что нарисовано на холсте"""
        self.canvas.delete("all")

    def classify_handwriting(self):
        """Вызывает функцию, которая классифицирует
           цифру и выводит результат"""
        pred, proc = self.get_answer()
        self.label.configure(text=str(pred)+', '+str(int(proc*100))+'%')

    def draw(self, event):
        """Реагирует на действие мышки и рисует
           овал радиуса 20, если обнаружено действие"""
        self.x = event.x
        self.y = event.y
        r = 20
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill='black')  # при нажатии делает овал

app = App()
mainloop()


