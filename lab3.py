from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	# изменить размер изображения до фиксированного размера, затем сгладить изображение в
	# список яркости сырых пикселей
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
  # извлекаем трехмерную гистограмму цвета из цветового пространства HSV, используя
	# указанное количество бинов на канал
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# обрабатываем нормализацию гистограммы если OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# иначе такая обработка
	else:
		cv2.normalize(hist, hist)

	# возвращаем сглаженную гистограмму как вектор признаков
	return hist.flatten()
  
# ввод через консоль данных
# создать аргумент, синтаксический анализ и анализ аргументов
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# список изображений, которые будем описывать
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

# инициализируем матрицу яркости сырых пикселей, матрицу функций,
# и список ярлыков
rawImages = []
features = []
labels = []

# циклом перебираем входные изображения
for (i, imagePath) in enumerate(imagePaths):
	# загружаем изображение и извлекаем метку класса название файла - {class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	# извлекаем необработанные "особенности" интенсивности пикселей, за которыми следует цвет
	# гистограмма для характеристики цветового распределения пикселей
	# на изображении
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# обновить необработанные изображения, функции и матрицы меток
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
# показывать обновление каждые 1000 изображений
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# разделить данные на части для обучения и тестирования
(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

# обучить и оценить классификатор k-NN по яркости необработанных пикселей
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# обучить и оценить классификатор k-NN на гистограмме
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
