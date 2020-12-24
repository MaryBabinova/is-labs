# С сайта Kaggle был загружен архив с фото собачками и кошками. Обучение производится на основе этих данных. Данные были распакованы и создана директория dataset_dogs_vs_cats
#с подкаталогами train и test. В каждом каталоге созданы подкаталоги cats и dogs. В каталог test было помещено около 25% изображений каждого вида.
#В примере взята архитектура с одним блоком vgg define_model() и архитектура с 2 блоками vgg define_model2(). И последующая проверка на производительность обоих методов для сравнения
#Алгоритм запускается толко один раз для каждого метода.

import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# define cnn model vgg1
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# define cnn model vgg2
def define_model2():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model 

# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    train_it = datagen.flow_from_directory('C:/Users/Мару/PycharmProjects/pythonProject/dataset_dogs_vs_cats/train/', class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = datagen.flow_from_directory('C:/Users/Мару/PycharmProjects/pythonProject/dataset_dogs_vs_cats/test/', class_mode='binary', batch_size=64, target_size=(200, 200))
    # fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))


# entry point, run the test harness
run_test_harness()


#Пример работы и вывод о производительности данной модели с одним блоком vgg
#Found 25000 images belonging to 2 classes.
#Found 6251 images belonging to 2 classes.
#> 90.514

#Пример работы и вывод о производительности данной модели с двумя блоками vgg
#Found 25000 images belonging to 2 classes.
#Found 6251 images belonging to 2 classes.
#>98.776
