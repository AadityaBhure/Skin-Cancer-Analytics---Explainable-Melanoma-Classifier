# Explainable skin-lesion classification

This academic mini project classifies HAM10000 dermoscopic images into **melanoma** and **non-melanoma** using ImageNet-pretrained EfficientNetB0 and Grad-CAM. EfficientNetB0 already contains squeeze-and-excitation channel-attention blocks. This is an educational prototype, not medical advice or a diagnostic system. Non-melanoma does **not** mean benign: this class includes basal cell carcinoma and other diagnoses.

## Run the already-configured project

**Double-click `Start Model.cmd` in this folder.** A basic desktop window opens; click **Choose image and predict** and select a JPG or PNG. The first prediction loads the model and may take a few seconds. No terminal commands or additional packages are needed. Close the window to quit.

The window shows the image prediction, both class scores, Grad-CAM, and saved test-set metrics. Its **Figure 4**, **Figure 5**, and **Figure 6** tabs contain training history, the confusion matrix, and the original/prediction/Grad-CAM comparison. Use **Save figure as…** to export each report image. Figure 6 updates after each prediction. Uploaded images stay local.

This is a binary model: Figure 5 includes both model classes, not all seven original HAM10000 diagnoses. A seven-class confusion matrix requires retraining a seven-class model. Test accuracy/precision/recall/F1/AUC describe the held-out dataset; they cannot be calculated from one unlabelled uploaded image.

### Optional terminal commands

The local environment and trained model are ready. From this `sourcecode` folder, paste only the commands inside the code block into PowerShell. Do not paste the Markdown backticks, the word `powershell`, or the terminal prompt. Do not recreate the existing `.venv`.

```powershell
$example = (Get-ChildItem data/processed/test/melanoma/*.jpg | Sort-Object Name | Select-Object -First 1).FullName
.\.venv\Scripts\python.exe src/predict.py --image "$example"
Invoke-Item outputs/figures/gradcam_example.png
```

For your own image, use `.\.venv\Scripts\python.exe src/predict.py --image "C:\path\to\image.jpg"`.

To rerun preparation, training and evaluation:

```powershell
.\.venv\Scripts\python.exe src/prepare_data.py
.\.venv\Scripts\python.exe src/train.py
.\.venv\Scripts\python.exe src/evaluate.py
```

## First-time setup on another machine

Use standard CPython 3.10–3.12. These setup commands are only for a fresh environment; do not run them over the configured environment above. On Windows with the Python launcher and Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m compileall src
```

Then run preparation, training and evaluation above. On macOS/Linux create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python`. Do not use Python 3.14 for this project's pinned TensorFlow range. Switching an existing environment's Python version leaves compiled packages incompatible; create environments with the intended interpreter from the start.

For NVIDIA GPUs, TensorFlow 2.15–2.17 requires Linux/WSL2; its native Windows package runs on CPU even when an RTX GPU is installed. See the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip#windows-native). The scripts automatically use a GPU when the TensorFlow runtime exposes one.

Manually place `HAM10000_metadata.csv` and both JPEG image folders under `data/raw/`. The supplied extensionless `HAM10000_metadata` is also accepted. Images are found recursively. No dataset API, account, key, backend or database is used. Installing packages and the **first ImageNet weights download** require internet; the weights are cached under `models/.keras/`. Subsequent runs work locally. CPU is supported automatically.

## Pipeline and outputs

1. `prepare_data.py`: checks image readability, samples at most 1,000 melanoma images and equally many other lesions, then makes seed-42 stratified 70/15/15 image splits. Only `data/processed/` is replaced. Raw inputs are preserved. `--sample-cap 500` is available if runtime is excessive; retrain and reevaluate after changing data.
2. `train.py`: 224×224 images, batch 16, frozen EfficientNetB0, global average pooling, dropout 0.30, sigmoid head, Adam 0.0001 and binary cross-entropy. Only training images receive flip/rotation/zoom augmentation. Runs at most eight epochs with early stopping and best-validation-loss checkpointing. No fine-tuning. EfficientNet's built-in rescaling expects pixels in 0–255; no extra division by 255 is applied.
3. `evaluate.py`: evaluates only the test split at threshold 0.50. Melanoma is the positive class. Class order is saved during training and checked during evaluation. Produces accuracy, melanoma precision/recall/F1, ROC-AUC, a classification report and image-level predictions.
4. `predict.py`: prints a single prediction and melanoma score, saves a Grad-CAM overlay and a three-panel report figure. Grad-CAM differentiates the melanoma probability through the last convolutional layer found programmatically. A zero positive heatmap is reported honestly.

The best model is `models/skin_cancer_efficientnetb0.keras`. Figures are under `outputs/figures/`: `dataset_distribution.png`, `training_curves.png`, `confusion_matrix.png`, `roc_curve.png`, `gradcam_example.png`. Metrics, class order, dataset summary and split manifest are under `outputs/metrics/`; the authoritative results are `test_metrics.json`. Predictions and the standalone Grad-CAM image are under `outputs/predictions/`.

## Interpretation and reproducibility

Report only the actual generated metrics, labelled **held-out balanced image-level test split**. The guide requires image-level stratified splitting. HAM10000 contains multiple images of some lesions, so lesion IDs can overlap across splits; `dataset_preparation.json` records these overlaps. Results are not lesion-independent or clinical validation. The balanced subset does not represent real-world prevalence, and sigmoid scores are not calibrated patient risk estimates. Grad-CAM shows regions influencing the model score, not lesion boundaries or medical proof.

Dataset selection and splits use seed 42; training seeds are fixed, but numeric results can vary across hardware and TensorFlow versions. Re-running preparation requires re-running training and evaluation. Raw images, processed images, runtime environments, models and generated outputs are ignored by Git.
