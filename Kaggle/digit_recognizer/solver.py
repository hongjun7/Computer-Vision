from warnings import simplefilter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
simplefilter("ignore")

train_data = pd.read_csv('C:\\Users\\hongj\\Desktop\\kaggle\\digit_recognizer\\train.csv')
train_labels = train_data['label']
train_labels = to_categorical(train_labels, num_classes=10)
train_data = np.array(train_data.drop('label', axis=1))
test_data = pd.read_csv('C:\\Users\\hongj\\Desktop\\kaggle\\digit_recognizer\\test.csv')
test_data = np.array(test_data)

train_images = train_data.reshape((42000, 28, 28, 1))
test_images = test_data.reshape((28000, 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images / 255.0

#train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=2, stratify=train_labels)
#print(train_images.shape, val_images.shape, train_labels.shape, val_labels.shape)

data_generator = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)
data_generator.fit(train_images)

model = models.Sequential([
    layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(data_generator.flow(train_images, train_labels, batch_size=86), epochs=30,
                              # validation_data=(val_images, val_labels),
                              verbose=2,
                              steps_per_epoch=train_images.shape[0]//86)
'''
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()
'''
predict = np.array(model.predict(test_images, verbose=2))
predict = np.argmax(predict, axis=1)
submission = pd.DataFrame()
submission['ImageId'] = [i for i in range(1, 28001)]
submission['Label'] = predict
submission.to_csv("C:\\Users\\hongj\\Desktop\\kaggle\\digit_recognizer\\submission.csv", index=False)
