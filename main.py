import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

#1 Pobranie danych cifar-10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

#2 Normalizacja danych

train_images, test_images = train_images / 255.0, test_images / 255.0

#3 Definicja modelu CNN

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

#4. Kompilacja modelu

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, #Zrobiłem 5, bo 10 to dużo czsu
                    validation_data=(test_images, test_labels))

#5 Testowanie modelu

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTEst accuracy:', test_acc)


