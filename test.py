import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = (180, 180)
IMAGE_PATH = 'data/dandelion/dandelion.8684108_a85764b22d_n.jpg'
MODEL_PATH = 'model/save_at_58.keras'

model = keras.models.load_model(MODEL_PATH)

img = keras.utils.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
results = [predictions[0][i] for i in range(0, len(predictions[0]))]
print(results)