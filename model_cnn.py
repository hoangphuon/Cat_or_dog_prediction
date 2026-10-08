import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from config import MODELS_DIR, EPOCHS_CNN, IMG_SIZE

def build_custom_cnn(img_size: int = IMG_SIZE, learning_rate: float = 1e-3) -> keras.Model:
	inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

	def conv_block(x, filters, drop=0.25):
		x = layers.Conv2D(filters, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
		x = layers.BatchNormalization()(x)
		x = layers.Activation("relu")(x)
		x = layers.MaxPooling2D((2, 2))(x)
		x = layers.Dropout(drop)(x)
		return x

	x = conv_block(inputs, 32, drop=0.2)
	x = conv_block(x, 64, drop=0.25)
	x = conv_block(x, 128, drop=0.3)
	x = conv_block(x, 256, drop=0.3)

	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)
	x = layers.Dropout(0.5)(x)

	outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

	model = keras.Model(inputs=inputs, outputs=outputs, name="DogCat_CNN")
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
		loss="binary_crossentropy",
		metrics=["accuracy"],
	)
	return model

def train_cnn(
	train_ds: tf.data.Dataset,
	val_ds: tf.data.Dataset,
	epochs: int = EPOCHS_CNN,
) -> tuple:
	model = build_custom_cnn()
	save_path = os.path.join(MODELS_DIR, "cnn_best.keras")

	cb_list = [
		callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
		callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
		callbacks.ModelCheckpoint(filepath=save_path, monitor="val_accuracy", save_best_only=True, verbose=0),
	]

	model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs,
		callbacks=cb_list,
		verbose=1,
	)

	results = model.evaluate(val_ds, verbose=0)
	val_acc = results[1]
	model.save(save_path)

	return model, val_acc

def predict_cnn(model: keras.Model, image_array: np.ndarray) -> dict:
	img = np.expand_dims(image_array, axis=0)
	prob_dog = float(model.predict(img, verbose=0)[0][0])
	prob_cat = 1.0 - prob_dog
	label = 1 if prob_dog >= 0.5 else 0

	return {
		"label": label,
		"class": "Dog" if label == 1 else "Cat",
		"confidence": prob_dog if label == 1 else prob_cat,
		"prob": {"Cat": prob_cat, "Dog": prob_dog}
	}