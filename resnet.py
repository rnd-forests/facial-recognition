import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import config
from shared import load_faces

epochs = 50
size = config.IMAGE_SIZE
n_classes = config.N_CLASSES
batch_size = config.BATCH_SIZE

X_train, X_test, y_train, y_test = load_faces("data_3_channels", channels=3)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

X_train = X_train.astype('float32')
X_train /= 255
X_train = X_train.reshape(X_train.shape[0], size, size, 3)

X_test = X_test.astype('float32')
X_test /= 255
X_test = X_test.reshape(X_test.shape[0], size, size, 3)

y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

base_model = InceptionResNetV2(include_top=False,
                               weights='imagenet',
                               input_shape=(size, size, 3),
                               pooling='max')

X = base_model.output
X = Dense(1024, activation="relu")(X)
X = Dropout(0.2)(X)
outputs = Dense(n_classes, activation="softmax")(X)

model = Model(inputs=base_model.input, outputs=outputs)

for layer in base_model.layers:
    layer.trainable = False

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
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])
