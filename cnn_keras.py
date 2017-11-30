import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D as BaseConv2D, MaxPool2D, Dense, Dropout, Flatten

import functools

import config
from shared import load_faces

# Test accuracy: 0.819569672131

epochs = 150
size = config.IMAGE_SIZE
n_classes = config.N_CLASSES
batch_size = config.BATCH_SIZE

X_train, X_test, y_train, y_test = load_faces("data_1_channels")

X_train = X_train.astype('float32')
X_train /= 255
X_train = X_train.reshape(X_train.shape[0], size, size, 1)

X_test = X_test.astype('float32')
X_test /= 255
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)


def partial_Conv2D(cls, *args, **kwargs):
    class PartialConv2D(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)
    return PartialConv2D


Conv2D = partial_Conv2D(BaseConv2D, padding="same", activation="elu",
                        kernel_initializer="he_normal", kernel_regularizer=l1_l2(0.02))

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(size, size, 1)))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(32, (3, 3)))
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
plot_model(model, to_file="./images/cnn_keras.png", show_shapes=True, show_layer_names=True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    steps_per_epoch=len(X_train) / batch_size,
                    verbose=1,
                    workers=4)

score = model.evaluate(X_test, y_test, verbose=1)

print('\n\nTest loss:', score[0])
print('Test accuracy:', score[1])
