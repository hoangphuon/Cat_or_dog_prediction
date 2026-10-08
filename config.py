import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test1")
MODELS_DIR = os.path.join(BASE_DIR, "models")

IMG_SIZE = 64
IMG_CHANNELS = 3
NUM_CLASSES = 2

RANDOM_SEED = 42
TEST_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS_DL = 15
EPOCHS_CNN = 15

os.makedirs(MODELS_DIR, exist_ok=True)