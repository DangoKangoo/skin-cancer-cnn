🩺 Skin Cancer CNN – Melanoma Detection (ISIC 2018)

Computer Vision (CP4210)

-This project builds a deep learning pipeline to detect melanoma vs non-melanoma using dermoscopy images from the ISIC 2018 dataset.

-The model uses MobileNetV2 transfer learning and is deployed through a clean Streamlit analytics dashboard.


⚠️ Educational use only. Not a medical device.


🚀 What This Project Does

-Converts ISIC 2018 multi-class labels → binary (melanoma vs rest)
-Creates reproducible train / validation / test splits
-Trains a MobileNetV2 transfer learning baseline
-Evaluates performance on a held-out test set
-Generates analytics + explainability dashboard:
-Confusion Matrix
-ROC / PR curves
-Calibration
-Error Explorer
-Grad-CAM
-Upload inference demo

🧰 Tech Stack

-Python 3.11
-PyTorch
-Torchvision
-Streamlit
-scikit-learn
-Matplotlib
-GPU optional. CUDA supported if available.

⚙️ Setup
From the project root:

1️⃣ Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate
2️⃣ Install Dependencies

If using CUDA 12.6:

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install streamlit scikit-learn matplotlib pandas pillow tqdm
(Adjust CUDA version if needed.)

📂 Dataset
Download ISIC 2018 Task 3:

Training Input images

Training GroundTruth CSV

Place them inside:

data/raw/images/ISIC2018_Task3_Training_Input/
data/raw/images/ISIC2018_Task3_Training_GroundTruth/

Do not rename folders.

🔬 Pipeline

1️⃣ Create Binary Labels + Splits
python src\data\make_dataset.py

Outputs:

data/splits/train.csv
data/splits/val.csv
data/splits/test.csv

Stratified split. Melanoma is the positive class.

2️⃣ Train Model
python src\train.py --epochs 5 --batch-size 32

Best checkpoint saved to:

models/best_mobilenetv2.pt
3️⃣ Evaluate on Test Set
python src\evaluate.py --ckpt models/best_mobilenetv2.pt

Outputs:

reports/metrics/test_metrics.json
reports/figures/test_confusion_matrix.png
4️⃣ Generate Dashboard Predictions
python src\predict_testset.py --ckpt models/best_mobilenetv2.pt --split data\splits\test.csv

Outputs:

reports/metrics/test_predictions.csv

(This file is auto-generated and ignored by git.)

5️⃣ Run the Dashboard
Make sure you are inside the virtual environment.

python -m streamlit run app\streamlit_app.py

Dashboard includes:
    Threshold tuning
    Confusion matrix
    ROC / PR curves
    Calibration analysis
    Error explorer
    Grad-CAM explainability
    Upload inference demo

📁 Project Structure
src/
  data/
  train.py
  evaluate.py
  predict_testset.py

app/
  core/
  ui/
  streamlit_app.py

data/
  raw/
  splits/

models/
reports/

⚠️ Limitations
-Trained only on ISIC dermoscopy images
-Performance may degrade on clinical smartphone photos (domain shift)
-Class imbalance (~11% melanoma)
-Not validated for real-world medical deployment
