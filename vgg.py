import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator

import config
from shared import load_preprocessed_faces

# Accuracy: 96.74%

epochs = 100
size = config.IMAGE_SIZE
n_classes = config.N_CLASSES
batch_size = config.BATCH_SIZE

X_train, X_test, y_train, y_test = load_preprocessed_faces("data_3_channels", channels=3)

vgg = VGGFace(include_top=False, weights='vggface', input_shape=(size, size, 3), pooling='max')

X = vgg.output
X = Dense(4096, activation="relu")(X)
X = Dropout(0.4)(X)
outputs = Dense(n_classes, activation="softmax")(X)

model = Model(inputs=vgg.input, outputs=outputs)

for layer in vgg.layers:
    layer.trainable = False

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
