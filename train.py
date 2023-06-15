import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = (180, 180)
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 100

DATA_PATH = 'data'
MODEL_PATH = 'model'


train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)


train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

def make_model(input_shape, NUM_CLASSES):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)

    conv_1 = layers.Conv2D(32, 3, strides=2, padding='same')(x)
    conv_1 = layers.MaxPooling2D(3, strides=2, padding="same")(conv_1)
    conv_1 = layers.Conv2D(64, 3, strides=2, padding='same')(conv_1)
    conv_1 = layers.MaxPooling2D(3, strides=2, padding="same")(conv_1)

    conv_2 = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    conv_2 = layers.MaxPooling2D(3, strides=2, padding="same")(conv_2)
    conv_2 = layers.Conv2D(128, 3, strides=2, padding='same')(conv_2)
    conv_2 = layers.MaxPooling2D(3, strides=2, padding="same")(conv_2)

    conv_3 = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    conv_3 = layers.MaxPooling2D(3, strides=2, padding="same")(conv_3)
    conv_3 = layers.Conv2D(265, 3, strides=2, padding="same")(conv_3)
    conv_3 = layers.MaxPooling2D(3, strides=2, padding="same")(conv_3)

    x = layers.Concatenate(axis=-1)([conv_1, conv_2, conv_3])
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if NUM_CLASSES == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = NUM_CLASSES

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=IMAGE_SIZE + (3,), NUM_CLASSES=NUM_CLASSES)

callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(MODEL_PATH, "save_at_{epoch}.keras")),
    keras.callbacks.EarlyStopping(monitor='loss',
                                patience=3,
                                verbose=1,
    ),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_ds,
    verbose=1,
)

