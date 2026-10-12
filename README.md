## Dogs vs Cats Classification Project

A project that builds and trains three artificial intelligence models (Machine Learning and Deep Learning) to classify images of dogs and cats:
1. An extended linear regression model (Logistic Regression / SGD Classifier)
2. A traditional Deep Learning model (Multilayer Perceptron - Dense Network)
3. A Convolutional Neural Network (CNN)

## 1. Directory structure

```
dogs_vs_cats/
|-- config.py              # Global configuration (image size, paths, epochs)
|-- data_loader.py         # Loads and preprocesses image data (NumPy arrays and tf.data.Dataset)
|-- model_linear.py        # Defines and predicts with the Logistic Regression / SGD model
|-- model_deep_learning.py # Defines, trains, and predicts with the Deep Learning (MLP) model
|-- model_cnn.py           # Defines, trains, and predicts with the CNN model
|-- train.py               # Script for running the complete training process
|-- predict.py             # Script for predicting any image
|-- README.md              # Usage instructions and operating principles
|-- models/                # Directory containing trained model files (.pkl and .keras)
`-- data/train/            # Directory containing 8,000 training images (cat.*.jpg and dog.*.jpg)
```

## 2. How each model works

### 2.1. Model 1: Logistic Regression / SGD Classifier (`model_linear.py`)
- **Input**: A `64 x 64 x 3` image flattened into a one-dimensional vector with `12,288` features.
- **Preprocessing**: The vector is passed through `StandardScaler` to normalize the data distribution to a mean of 0 and a standard deviation of 1.
- **Operation**: Uses logistic regression with a sigmoid activation function to calculate the probability of binary classification (0 = Cat, 1 = Dog). The model uses the `scikit-learn` library.
- **Characteristics**: Very fast training (under 1 minute), making it suitable as a baseline model. However, it does not capture the 2D spatial relationships between image pixels well.

### 2.2. Model 2: Deep Learning MLP (`model_deep_learning.py`)
- **Input**: A one-dimensional vector with `12,288` elements.
- **Network architecture**:
  - `Dense(512)` -> `BatchNormalization` -> `ReLU` -> `Dropout(0.4)`
  - `Dense(256)` -> `BatchNormalization` -> `ReLU` -> `Dropout(0.4)`
  - `Dense(128)` -> `BatchNormalization` -> `ReLU` -> `Dropout(0.3)`
  - `Dense(64)`  -> `BatchNormalization` -> `ReLU` -> `Dropout(0.2)`
  - `Dense(1, activation='sigmoid')` output layer
- **How it works**:
  - `BatchNormalization` stabilizes gradients and accelerates convergence across epochs.
  - `Dropout` and `L2` regularization help prevent overfitting.
  - `EarlyStopping` automatically restores the best weights from the validation set.
- **Characteristics**: Learns more complex nonlinear relationships than traditional linear regression.

### 2.3. Model 3: Convolutional Neural Network - CNN (`model_cnn.py`)
- **Input**: The original three-dimensional image matrix `(64, 64, 3)`.
- **Network architecture**:
  - Block 1: `Conv2D(32, 3x3)` -> `BatchNorm` -> `ReLU` -> `MaxPooling2D(2x2)` -> `Dropout(0.2)`
  - Block 2: `Conv2D(64, 3x3)` -> `BatchNorm` -> `ReLU` -> `MaxPooling2D(2x2)` -> `Dropout(0.25)`
  - Block 3: `Conv2D(128, 3x3)` -> `BatchNorm` -> `ReLU` -> `MaxPooling2D(2x2)` -> `Dropout(0.3)`
  - Block 4: `Conv2D(256, 3x3)` -> `BatchNorm` -> `ReLU` -> `MaxPooling2D(2x2)` -> `Dropout(0.3)`
  - Classification: `GlobalAveragePooling2D` -> `Dense(256)` -> `Dropout(0.5)` -> `Dense(1, sigmoid)`
- **How it works**:
  - The `Conv2D` layers apply convolution operations to extract spatial features such as edges, corners, fur textures, faces, and the ears of dogs and cats.
  - `GlobalAveragePooling2D` significantly reduces the number of parameters compared with `Flatten`, lowering the risk of overfitting and speeding up computation.

---

## 3. Usage instructions

### 3.1. Predicting an image (`predict.py`)
Predict results directly in the Terminal (fast and concise, without opening a chart interface):

```powershell
# Predict one image with all 3 models at the same time:
python predict.py --image "data\train\dog.20.jpg" --all

# Predict using only the CNN model:
python predict.py --image "data\train\cat.25.jpg" --model cnn

# Predict using the Deep Learning (MLP) model:
python predict.py --image "data\train\cat.25.jpg" --model dl

# Predict using the Linear Regression model:
python predict.py --image "data\train\dog.20.jpg" --model linear
```

**Example output format:**
```text
==================================================
IMAGE: dog.20.jpg
==================================================
  [Linear (Logistic)   ]: Dog (71.47%) | P(Cat): 0.29, P(Dog): 0.71
  [Deep Learning (MLP) ]: Dog (65.01%) | P(Cat): 0.35, P(Dog): 0.65
  [CNN                 ]: Dog (50.92%) | P(Cat): 0.49, P(Dog): 0.51
==================================================
```

### 3.2. Retraining the models (`train.py`)
To retrain the models with a different number of images or epochs:

```powershell
# Train all 3 models with 3,000 images (default):
python train.py --samples 3000

# Train only the CNN model:
python train.py --models cnn --samples 3000 --epochs-cnn 15

# Train only the Linear and Deep Learning models with 2,000 images:
python train.py --models linear dl --samples 2000 --epochs-dl 10
```

**Training results are printed concisely in the Terminal:**
```text
==================================================
RESULTS (ACCURACY)
==================================================
  CNN                        : 0.5525 (55.25%)
  Logistic Regression        : 0.5375 (53.75%)
  SGD Classifier             : 0.5225 (52.25%)
  Deep Learning (MLP)        : 0.5167 (51.67%)
==================================================
Total time: 2.15 minutes
```

## 4. Model files in the `models/` directory
- `logistic_regression.pkl`: Trained Logistic Regression model.
- `sgd_classifier.pkl`: Trained SGD Classifier model.
- `dl_mlp_best.keras`: Best trained Deep Learning MLP model.
- `cnn_best.keras`: Best trained CNN model.
