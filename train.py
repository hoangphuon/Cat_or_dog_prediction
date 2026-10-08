import argparse
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import TRAIN_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS_DL, EPOCHS_CNN, MODELS_DIR
from data_loader import load_data_numpy, build_tf_dataset
from model_linear import train_logistic_regression, train_sgd_classifier
from model_deep_learning import train_deep_learning
from model_cnn import train_cnn

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--models", nargs="+", choices=["linear", "dl", "cnn"], default=["linear", "dl", "cnn"])
	parser.add_argument("--samples", type=int, default=3000)
	parser.add_argument("--epochs-dl", type=int, default=EPOCHS_DL)
	parser.add_argument("--epochs-cnn", type=int, default=EPOCHS_CNN)
	return parser.parse_args()

def main():
	args = parse_args()

	if not os.path.exists(TRAIN_DIR):
		print(f"Error: {TRAIN_DIR} not found")
		sys.exit(1)

	results = {}
	start_time = time.time()

	if any(m in args.models for m in ["linear", "dl"]):
		X_train, X_val, y_train, y_val = load_data_numpy(
			img_size=IMG_SIZE,
			max_samples=args.samples,
		)

		if "linear" in args.models:
			_, acc_log = train_logistic_regression(X_train, y_train, X_val, y_val)
			_, acc_sgd = train_sgd_classifier(X_train, y_train, X_val, y_val)
			results["Logistic Regression"] = acc_log
			results["SGD Classifier"] = acc_sgd

		if "dl" in args.models:
			_, acc_dl = train_deep_learning(
				X_train, y_train, X_val, y_val,
				epochs=args.epochs_dl,
				batch_size=BATCH_SIZE,
			)
			results["Deep Learning (MLP)"] = acc_dl

	if "cnn" in args.models:
		train_ds, val_ds, _, _ = build_tf_dataset(
			img_size=IMG_SIZE,
			batch_size=BATCH_SIZE,
			max_samples=args.samples,
		)
		_, acc_cnn = train_cnn(
			train_ds, val_ds,
			epochs=args.epochs_cnn,
		)
		results["CNN"] = acc_cnn

	print("\n" + "=" * 50)
	print("RESULTS (ACCURACY)")
	print("=" * 50)
	for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
		print(f"  {model_name:<26} : {acc:.4f} ({acc * 100:.2f}%)")
	print("=" * 50)
	print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
	main()