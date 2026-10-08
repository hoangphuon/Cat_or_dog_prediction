import argparse
import os
import warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import IMG_SIZE, MODELS_DIR
from model_linear import predict_linear
from model_deep_learning import predict_dl
from model_cnn import predict_cnn

def load_image(image_path: str, img_size: int = IMG_SIZE) -> np.ndarray:
	if not os.path.exists(image_path):
		raise FileNotFoundError(image_path)

	img = cv2.imread(image_path)
	if img is None:
		raise ValueError(image_path)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (img_size, img_size))
	return img.astype(np.float32) / 255.0

def predict_single_model(model_type: str, image: np.ndarray) -> dict:
	if model_type == "linear":
		import joblib
		path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
		if not os.path.exists(path):
			path = os.path.join(MODELS_DIR, "sgd_classifier.pkl")
		if not os.path.exists(path):
			return {"error": "Linear model not found"}
		model = joblib.load(path)
		return predict_linear(model, image)

	elif model_type == "dl":
		import tensorflow as tf
		for fname in ["dl_mlp_best.keras", "dl_mlp_final.keras"]:
			path = os.path.join(MODELS_DIR, fname)
			if os.path.exists(path):
				model = tf.keras.models.load_model(path)
				return predict_dl(model, image)
		return {"error": "DL model not found"}

	elif model_type == "cnn":
		import tensorflow as tf
		for fname in ["cnn_best.keras", "cnn_CustomCNN_best.keras", "cnn_CustomCNN_final.keras"]:
			path = os.path.join(MODELS_DIR, fname)
			if os.path.exists(path):
				model = tf.keras.models.load_model(path)
				return predict_cnn(model, image)
		return {"error": "CNN model not found"}

	return {"error": f"Invalid model: {model_type}"}

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--image", required=True)
	parser.add_argument("--model", choices=["linear", "dl", "cnn"], default="cnn")
	parser.add_argument("--all", action="store_true")
	args = parser.parse_args()

	image = load_image(args.image)
	print("\n" + "=" * 50)
	print(f"IMAGE: {os.path.basename(args.image)}")
	print("=" * 50)

	if args.all:
		models = [
			("Linear (Logistic)", "linear"),
			("Deep Learning (MLP)", "dl"),
			("CNN", "cnn"),
		]
		for name, mtype in models:
			res = predict_single_model(mtype, image)
			if "error" in res:
				print(f"  [{name}]: {res['error']}")
			else:
				print(f"  [{name:<20}]: {res['class']} ({res['confidence']:.2%}) | P(Cat): {res['prob']['Cat']:.2f}, P(Dog): {res['prob']['Dog']:.2f}")
	else:
		res = predict_single_model(args.model, image)
		if "error" in res:
			print(f"  Error: {res['error']}")
		else:
			print(f"  Prediction : {res['class']}")
			print(f"  Confidence : {res['confidence']:.2%}")
			print(f"  P(Cat)     : {res['prob']['Cat']:.2%}")
			print(f"  P(Dog)     : {res['prob']['Dog']:.2%}")

	print("=" * 50 + "\n")

if __name__ == "__main__":
	main()