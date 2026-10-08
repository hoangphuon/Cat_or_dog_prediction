import os
import re
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import TRAIN_DIR, IMG_SIZE, RANDOM_SEED, TEST_SPLIT

def _label_from_filename(filename: str) -> int:
	name = filename.lower()
	if name.startswith("cat"):
		return 0
	elif name.startswith("dog"):
		return 1
	raise ValueError(filename)


def _load_image_cv2(path: str, size: int = IMG_SIZE) -> np.ndarray:
	img = cv2.imread(path)
	if img is None:
		return None
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (size, size))
	return img

def load_data_numpy(
	train_dir: str = TRAIN_DIR,
	img_size: int = IMG_SIZE,
	max_samples: int = None,
	normalize: bool = True,
) -> tuple:
	if not os.path.exists(train_dir):
		raise FileNotFoundError(train_dir)

	files = sorted(os.listdir(train_dir))
	files = [f for f in files if re.search(r"\.(jpg|jpeg|png)$", f, re.I)]

	if max_samples:
		cats = [f for f in files if f.startswith("cat")][:max_samples // 2]
		dogs = [f for f in files if f.startswith("dog")][:max_samples // 2]
		files = cats + dogs

	images, labels = [], []
	for fname in tqdm(files, desc="Loading data"):
		path = os.path.join(train_dir, fname)
		img = _load_image_cv2(path, img_size)
		if img is not None:
			images.append(img)
			labels.append(_label_from_filename(fname))

	X = np.array(images, dtype=np.float32)
	y = np.array(labels, dtype=np.int32)

	if normalize:
		X /= 255.0

	return train_test_split(
		X, y,
		test_size=TEST_SPLIT,
		random_state=RANDOM_SEED,
		stratify=y,
	)

def _parse_image(file_path: tf.Tensor, label: tf.Tensor, img_size: int):
	raw = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(raw, channels=3)
	img = tf.image.resize(img, [img_size, img_size])
	img = tf.cast(img, tf.float32) / 255.0
	return img, label

def _augment(img: tf.Tensor, label: tf.Tensor):
	img = tf.image.random_flip_left_right(img)
	img = tf.image.random_brightness(img, max_delta=0.15)
	img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
	img = tf.clip_by_value(img, 0.0, 1.0)
	return img, label

def build_tf_dataset(
	train_dir: str = TRAIN_DIR,
	img_size: int = IMG_SIZE,
	batch_size: int = 32,
	max_samples: int = None,
	augment_train: bool = True,
) -> tuple:
	if not os.path.exists(train_dir):
		raise FileNotFoundError(train_dir)

	files = sorted(os.listdir(train_dir))
	files = [f for f in files if re.search(r"\.(jpg|jpeg|png)$", f, re.I)]

	if max_samples:
		cats = [f for f in files if f.startswith("cat")][:max_samples // 2]
		dogs = [f for f in files if f.startswith("dog")][:max_samples // 2]
		files = cats + dogs

	paths = [os.path.join(train_dir, f) for f in files]
	labels = [_label_from_filename(f) for f in files]

	p_tr, p_val, l_tr, l_val = train_test_split(
		paths, labels,
		test_size=TEST_SPLIT,
		random_state=RANDOM_SEED,
		stratify=labels,
	)

	def _make_ds(ps, ls, augment=False):
		ds = tf.data.Dataset.from_tensor_slices((ps, tf.cast(ls, tf.int32)))
		ds = ds.map(
			lambda p, l: _parse_image(p, l, img_size),
			num_parallel_calls=tf.data.AUTOTUNE,
		)
		if augment:
			ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
		ds = ds.cache()
		if augment:
			ds = ds.shuffle(buffer_size=2000, seed=RANDOM_SEED)
		ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
		return ds

	train_ds = _make_ds(p_tr, l_tr, augment=augment_train)
	val_ds = _make_ds(p_val, l_val, augment=False)

	return train_ds, val_ds, len(p_tr), len(p_val)