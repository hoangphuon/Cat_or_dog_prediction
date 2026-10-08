import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from config import MODELS_DIR, EPOCHS_DL, BATCH_SIZE

def build_deep_learning_model(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
	inputs = keras.Input(shape=(input_dim,), name="input_flat")

	x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)
	x = layers.Dropout(0.4)(x)

	x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)
	x = layers.Dropout(0.4)(x)

	x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)
	x = layers.Dropout(0.3)(x)

	x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)
	x = layers.Dropout(0.2)(x)

	outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

	model = keras.Model(inputs=inputs, outputs=outputs, name="DogCat_MLP")
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
		loss="binary_crossentropy",
		metrics=["accuracy"],
	)
	return model

def train_deep_learning(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_val: np.ndarray,
	y_val: np.ndarray,
	epochs: int = EPOCHS_DL,
	batch_size: int = BATCH_SIZE,
) -> tuple:
	X_tr_flat = X_train.reshape(X_train.shape[0], -1)
	X_val_flat = X_val.reshape(X_val.shape[0], -1)
	input_dim = X_tr_flat.shape[1]

	model = build_deep_learning_model(input_dim)
	save_path = os.path.join(MODELS_DIR, "dl_mlp_best.keras")

	cb_list = [
		callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
		callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
		callbacks.ModelCheckpoint(filepath=save_path, monitor="val_accuracy", save_best_only=True, verbose=0),
	]

	model.fit(
		X_tr_flat, y_train,
		validation_data=(X_val_flat, y_val),
		epochs=epochs,
		batch_size=batch_size,
		callbacks=cb_list,
		verbose=1,
	)

	results = model.evaluate(X_val_flat, y_val, verbose=0)
	val_acc = results[1]
	model.save(save_path)

	return model, val_acc

def predict_dl(model: keras.Model, image_array: np.ndarray) -> dict:
	flat = image_array.flatten().reshape(1, -1)
	prob_dog = float(model.predict(flat, verbose=0)[0][0])
	prob_cat = 1.0 - prob_dog
	label = 1 if prob_dog >= 0.5 else 0

	return {
		"label": label,
		"class": "Dog" if label == 1 else "Cat",
		"confidence": prob_dog if label == 1 else prob_cat,
		"prob": {"Cat": prob_cat, "Dog": prob_dog}
	}