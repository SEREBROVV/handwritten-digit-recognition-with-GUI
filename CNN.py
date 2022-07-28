import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # загружаем тестовую выборку и тренировочную
x_train = x_train.astype('float32')  # меняем тип элемента в списке из int в float
x_test = x_test.astype('float32')  # меняем тип элемента в списке из int в float

x_train = x_train.reshape(60000, 28, 28, 1)  # изменил размерность списка
x_test = x_test.reshape(10000, 28, 28, 1)  # изменил размарность списка

x_train /= 255  # поделили каждое значение в спиcке на 255
x_test /= 255  # поделили каждое значение в спиcке на 255
print(y_train[0])
print('==================')
y_train = to_categorical(y_train)  # преобразовали числа в массив (3 -> [0,0,0,1,0,0,0,0,0,0,0])
y_test = to_categorical(y_test)

model = Sequential() # создали модель с пустым cтеком слоёв(layers)
''' Добавляем в модель первый сверточный слой: 
    с 6 картами признаков(наборами нейронов, 
    которые находятся на плоскости и сворачивают
    изображение). Данный слой подразумевает создание
    ядра - матрицы весов, с помощью которых сворачивается
    изображение. input_shape - размерность картинки(в нашем
    случае она черно-белая). activation - функция активации
    через которую проходит сверточное изображение.
    padding = 'same' означает, что вход дынные и выходные данные
    после прохода через этот слой будут иметь одинаковую размерность'''
model.add(Conv2D(6, 5, input_shape=(28, 28, 1), activation='relu', padding='same'))
model.add(MaxPooling2D())  # пуллинговый слой, выбирается максимальные элемент(группа пикселей уплотняются до одного пикселя)
model.add(Conv2D(16, 5, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))  # исключили часть нейронов, чтобы не было переобучения
model.add(Flatten())  # конвертируем входные данные в меньшую размерность
model.add(Dense(120, activation='relu'))  # обычный слой персептрона
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))  # выходной слой

'''Собираем модель и выбираем у неё функцию потерь, оптимизатор и метрику.
   Фукнция потерь: задача её минимизировать, она определяет разность между
   ответом, который получила нейросеть и ответом который должен быть на
   самом деле.
   Оптимизатор - путь обучения нейросети.'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

'''Обучение нейросети
   x, y - данные на которых сеть учится
   validation_data - данные для оценки ошибки
   epochs - кол-во эпох(эпоха - итерация по
   всем слоям модели один раз с входными данными)
   batch_size - общее число тренировочных объектов,
   представленных в одном батче. Батч - объём данных,
   проходимых через нейросеть за одну итерацию
   (проходимых полностью через модель).
   '''
h = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=50)

'''Изображаем процент правильных ответов на каждой эпохе'''
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

'''Изображаем функцию потерь'''
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

model.save('model_mnist.h5')