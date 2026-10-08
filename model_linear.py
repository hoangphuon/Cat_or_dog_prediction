import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from config import MODELS_DIR

def flatten_images(X: np.ndarray) -> np.ndarray:
	return X.reshape(X.shape[0], -1)

def train_logistic_regression(X_train, y_train, X_val, y_val):
	X_tr_flat = flatten_images(X_train)
	X_val_flat = flatten_images(X_val)

	model = Pipeline([
		("scaler", StandardScaler()),
		("clf", LogisticRegression(
			C=0.01,
			max_iter=1000,
			random_state=42,
			n_jobs=-1
		)),
	])

	model.fit(X_tr_flat, y_train)
	y_pred = model.predict(X_val_flat)
	acc = accuracy_score(y_val, y_pred)

	save_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
	joblib.dump(model, save_path)
	return model, acc

def train_sgd_classifier(X_train, y_train, X_val, y_val):
	X_tr_flat = flatten_images(X_train)
	X_val_flat = flatten_images(X_val)

	model = Pipeline([
		("scaler", StandardScaler()),
		("clf", SGDClassifier(
			loss="log_loss",
			penalty="l2",
			alpha=0.001,
			max_iter=100,
			random_state=42,
			n_jobs=-1
		)),
	])

	model.fit(X_tr_flat, y_train)
	y_pred = model.predict(X_val_flat)
	acc = accuracy_score(y_val, y_pred)

	save_path = os.path.join(MODELS_DIR, "sgd_classifier.pkl")
	joblib.dump(model, save_path)
	return model, acc

def predict_linear(model, image_array: np.ndarray) -> dict:
	flat = image_array.flatten().reshape(1, -1)
	pred = int(model.predict(flat)[0])

	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(flat)[0]
		confidence = float(proba[pred])
		p_cat, p_dog = float(proba[0]), float(proba[1])
	else:
		confidence = 1.0
		p_cat, p_dog = (0.0, 1.0) if pred == 1 else (1.0, 0.0)

	return {
		"label": pred,
		"class": "Dog" if pred == 1 else "Cat",
		"confidence": confidence,
		"prob": {"Cat": p_cat, "Dog": p_dog}
	}