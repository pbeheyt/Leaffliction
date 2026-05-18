# Leaffliction

## Short Description
Leaffliction is a computer vision pipeline to analyze and classify leaf diseases from images (apple and grape classes). It covers dataset analysis, augmentation, transformation, training, and prediction.

## Architecture
- `src/data_analysis/`
  - Dataset distribution charts
  - Data augmentation and class balancing
- `src/data_transformation/`
  - Image transformations (mask, blur, ROI, landmarks, histogram)
- `src/classification/`
  - Feature extraction
  - Training (`train.py`) with validation metrics and saved model
  - Prediction (`predict.py`) with confidence and visualization
- `data/`
  - Source images and augmented/transformed outputs
- `results/`
  - Generated charts, reports, and prediction visualizations
- `models/`
  - Trained model artifact (`leaf_model.joblib`)

## Requirements
- Linux/macOS
- Python 3.10+
- Packages from `requirements.txt` (OpenCV, scikit-learn, matplotlib, numpy, etc.)

## Setup
```bash
make create-venv
make install-deps
```

## Run
All commands are run from project root.

### 1) Distribution
```bash
make distribution src=data/leaves/images
```
Generates class distribution charts.

### 2) Augmentation
Single image augmentation:
```bash
make augmentation img="data/leaves/images/Apple_Black_rot/image (1).JPG"
```
Dataset balancing (augmented output in `data/augmented`):
```bash
make balance src=data/leaves/images
```

### 3) Train
```bash
make train src=data/augmented
```
Outputs:
- `models/leaf_model.joblib`
- `results/classification/validation_report.json`
- `results/classification/validation_report.txt`

### 4) Predict
```bash
make predict img="data/leaves/images/Apple_Black_rot/image (1).JPG"
```
Prints predicted class + confidence and saves visualization in `results/prediction/`.

## Evaluation
Create the submission archive and SHA1 signature:

```bash
make package
make signature
```

Generated files:
- `packages/dataset.zip`
- `signature.txt`
