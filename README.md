## Dogs vs Cats Classification Project

Du an xay dung va huan luyen 3 mo hinh tri tue nhan tao (Machine Learning & Deep Learning) de phan loai hinh anh Cho va Meo:
1. Mo hinh Hoi quy tuyen tinh mo rong (Logistic Regression / SGD Classifier)
2. Mo hinh Deep Learning truyen thong (Multilayer Perceptron - Dense Network)
3. Mo hinh Mang no-ron tich chap (Convolutional Neural Network - CNN)

## 1. Cau truc thu muc

```
dogs_vs_cats/
|-- config.py              # Thiet lap thong so toan cuc (kich thuoc anh, duong dan, epochs)
|-- data_loader.py         # Doc va tien xu ly du lieu anh (numpy array va tf.data.Dataset)
|-- model_linear.py        # Dinh nghia va ham du doan cua mo hinh Logistic Regression / SGD
|-- model_deep_learning.py # Dinh nghia, huan luyen va du doan cua mo hinh Deep Learning (MLP)
|-- model_cnn.py           # Dinh nghia, huan luyen va du doan cua mo hinh Mang tich chap CNN
|-- train.py               # Script chay huan luyen tong hop
|-- predict.py             # Script chay du doan hinh anh bat ky
|-- README.md              # Huong dan su dung va nguyen ly hoat dong
|-- models/                # Thu muc chua cac file mo hinh da huan luyen (.pkl va .keras)
`-- data/train/            # Thu muc chua 8.000 anh huan luyen (cat.*.jpg va dog.*.jpg)
```

## 2. Nguyen ly hoat dong cua tung mo hinh

### 2.1. Mo hinh 1: Logistic Regression / SGD Classifier (`model_linear.py`)
- **Dau vao**: Anh kich thuoc `64 x 64 x 3` duoc trai phang (flatten) thanh vector 1 chieu co do dai `12.288` dac trung.
- **Tien xu ly**: Vector duoc dua qua `StandardScaler` de chuan hoa phan phoi du lieu ve trung binh 0 va do lech chuan 1.
- **Hoat dong**: Su dung ham hoi quy logistic voi ham kich hoat sigmoid de tinh toan xac suat phan lop nhi phan (0 = Cat, 1 = Dog). Mo hinh su dung thu vien `scikit-learn`.
- **Dac diem**: Huan luyen cuc nhanh (duoi 1 phut), phu hop lam mo hinh co so (baseline), tuy nhien de bo sot quan he khong gian 2D giua cac diem anh.

### 2.2. Mo hinh 2: Deep Learning MLP (`model_deep_learning.py`)
- **Dau vao**: Vector 1 chieu `12.288` phan tu.
- **Kien truc mang**:
  - `Dense(512)` -> `BatchNormalization` -> `ReLU` -> `Dropout(0.4)`
  - `Dense(256)` -> `BatchNormalization` -> `ReLU` -> `Dropout(0.4)`
  - `Dense(128)` -> `BatchNormalization` -> `ReLU` -> `Dropout(0.3)`
  - `Dense(64)`  -> `BatchNormalization` -> `ReLU` -> `Dropout(0.2)`
  - `Dense(1, activation='sigmoid')` o dau ra
- **Co che hoat dong**:
  - `BatchNormalization` giup on dinh gradient va tang toc do hoi tu qua cac epoch.
  - `Dropout` va chuan hoa `L2` giup chong hien tuong hoc vet (overfitting).
  - Tich hop `EarlyStopping` tu dong phuc hoi bo trong so tot nhat tren tap validation.
- **Dac diem**: Hoc duoc cac moi quan he phi tuyen tinh phuc tap hon so voi hoi quy tuyen tinh truyen thong.

### 2.3. Mo hinh 3: Convolutional Neural Network - CNN (`model_cnn.py`)
- **Dau vao**: Ma tran anh 3 chieu nguyen ban `(64, 64, 3)`.
- **Kien truc mang**:
  - Block 1: `Conv2D(32, 3x3)` -> `BatchNorm` -> `ReLU` -> `MaxPooling2D(2x2)` -> `Dropout(0.2)`
  - Block 2: `Conv2D(64, 3x3)` -> `BatchNorm` -> `ReLU` -> `MaxPooling2D(2x2)` -> `Dropout(0.25)`
  - Block 3: `Conv2D(128, 3x3)` -> `BatchNorm` -> `ReLU` -> `MaxPooling2D(2x2)` -> `Dropout(0.3)`
  - Block 4: `Conv2D(256, 3x3)` -> `BatchNorm` -> `ReLU` -> `MaxPooling2D(2x2)` -> `Dropout(0.3)`
  - Phan loai: `GlobalAveragePooling2D` -> `Dense(256)` -> `Dropout(0.5)` -> `Dense(1, sigmoid)`
- **Co che hoat dong**:
  - Cac lop `Conv2D` ap dung phep quet tich chap de trich xuat dac trung khong gian: canh, goc, ket cau long, mat, tai cua cho va meo.
  - `GlobalAveragePooling2D` giup giam thieu so luong tham so dang ke so voi `Flatten`, giam nguy co overfitting va tang toc do tinh toan.

---

## 3. Huong dan su dung

### 3.1. Du doan hinh anh (`predict.py`)
Du doan ket qua truc tiep ra man hinh Terminal (nhanh, gon, khong can mo giao dien bieu do):

```powershell
# Du doan 1 buc anh bang ca 3 mo hinh cung luc:
python predict.py --image "data\train\dog.20.jpg" --all

# Du doan bang rieng mo hinh CNN:
python predict.py --image "data\train\cat.25.jpg" --model cnn

# Du doan bang mo hinh Deep Learning (MLP):
python predict.py --image "data\train\cat.25.jpg" --model dl

# Du doan bang mo hinh Hoi quy tuyen tinh:
python predict.py --image "data\train\dog.20.jpg" --model linear
```

**Dinh dang ket qua tra ve tren man hinh:**
```text
==================================================
IMAGE: dog.20.jpg
==================================================
  [Linear (Logistic)   ]: Dog (71.47%) | P(Cat): 0.29, P(Dog): 0.71
  [Deep Learning (MLP) ]: Dog (65.01%) | P(Cat): 0.35, P(Dog): 0.65
  [CNN                 ]: Dog (50.92%) | P(Cat): 0.49, P(Dog): 0.51
==================================================
```

### 3.2. Huan luyen lai mo hinh (`train.py`)
Khi ban muon huan luyen lai voi so luong anh hoac so epoch khac:

```powershell
# Huan luyen ca 3 mo hinh voi 3.000 anh (mac dinh):
python train.py --samples 3000

# Chi huan luyen rieng mo hinh CNN:
python train.py --models cnn --samples 3000 --epochs-cnn 15

# Chi huan luyen mo hinh Linear va Deep Learning voi 2.000 anh:
python train.py --models linear dl --samples 2000 --epochs-dl 10
```

**Ket qua sau khi huan luyen se duoc in gon gang ra man hinh:**
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

## 4. Danh sach file mo hinh trong thu muc `models/`
- `logistic_regression.pkl`: Mo hinh Logistic Regression da huan luyen.
- `sgd_classifier.pkl`: Mo hinh SGD Classifier da huan luyen.
- `dl_mlp_best.keras`: Mo hinh Deep Learning MLP da huan luyen tot nhat.
- `cnn_best.keras`: Mo hinh Mang tich chap CNN da huan luyen tot nhat.