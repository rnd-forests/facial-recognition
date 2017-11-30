import keras
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D as BaseConv2D, MaxPool2D, Dense, Dropout, Flatten

import functools

import config
from shared import load_preprocessed_faces

# Test accuracy: 0.819569672131

epochs = 100
size = config.IMAGE_SIZE
n_classes = config.N_CLASSES
batch_size = config.BATCH_SIZE

X_train, X_test, y_train, y_test = load_preprocessed_faces("data_1_channels")


def partial_Conv2D(cls, *args, **kwargs):
    class PartialConv2D(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)
    return PartialConv2D


Conv2D = partial_Conv2D(
    BaseConv2D,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    kernel_regularizer=l1_l2(0.02))

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(size, size, 1)))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(128, (3, 3)))
model.add(MaxPool2D(2, 3))

model.add(Flatten())
model.add(Dense(1024, activation="elu"))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))

model.summary()

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=120,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(X_train)

model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=len(X_train) / batch_size,
    verbose=1,
    workers=4)

score = model.evaluate(X_test, y_test, verbose=1)

print('\n\nTest loss:', score[0])
print('Test accuracy:', score[1])
