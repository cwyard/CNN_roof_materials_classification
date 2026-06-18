# Rooftop material classification using CNN ensembles

This repository provides the complete implementation used to investigate multi-class rooftop material classification from very high-resolution (VHR) aerial imagery using convolutional neural networks (CNNs) and ensemble learning. The framework includes data preprocessing, optional image enhancement techniques (CLAHE and HOG), optional data augmentation, custom CNN architectures, optional transfer learning with MobileNetV2 pre-trained on ImageNet, configurable ensemble sizes, and optional class imbalance mitigation through weighted categorical cross-entropy.

The workflow was developed to assess deep learning performance under challenging conditions, including limited training data (1,600 annotated RGB image patches at 5-cm spatial resolution), a relatively high number of roofing material classes (12 classes), and substantial intra-class variability.

The repository aims to facilitate reproducibility and support further methodological developments in roof material mapping from aerial imagery.


## Cite this code

Please use the following DOI for citing this code:  https://doi.org/10.5281/zenodo.20730576

## Repository structure

```text
notebooks/
  DL3_training.ipynb      # Reproducible training workflow
src/
  config.py               # Paths and experiment parameters
  data.py                 # Data loading, filtering and augmentation
  preprocessing.py        # HOG-based rotation of anisotropic patches
  model.py                # CNN architecture
  training.py             # Single-model and ensemble training
  evaluation.py           # Metrics and figures
  utils.py                # Reproducibility and output helpers
trained_models/           # Pre-trained models in *.h5 format
  model1.h5
  model2.h5
  ...
  model12.h5
  README.md               # Model configuration
```
## Environment

The experiments were developed and tested using:

- Python 3.10
- TensorFlow 2.9.1
- Keras 2.9.0

The complete list of required packages is provided in `requirements.txt`.

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Before running

Edit `src/config.py` and update:

- `DATA_DIR`, which must contain the `red/`, `green/` and `blue/` folders;
- `OUTPUT_BASE_DIR`, where trained models and figures will be saved;
- the training parameters, if needed.

## Running the workflow

Open and run:

```bash
jupyter notebook notebooks/DL3_training.ipynb
```

The notebook loads the image patches, applies the pre-processing steps, trains the CNN ensemble, evaluates the predictions and saves the trained models and figures.

## Data availability

The annotated dataset of 1600 images used for the training and evaluation of the classifiers can be found here: https://doi.org/10.5281/zenodo.10406843

The expected data organisation is:

```text
data/
  red/*.tif
  green/*.tif
  blue/*.tif
```

File names are expected to contain the patch identifier and the class label, following the convention used in the manuscript dataset.

## Pre-trained models

The repository includes the 12 pre-trained CNN models used for the ensemble-learning experiments reported in the manuscript. These models were trained using the reference configuration consisting of CLAHE enhancement, HOG-based orientation normalisation, data augmentation, and standard categorical cross-entropy loss (i.e. without class weighting).

The models are provided in HDF5 (.h5) format and can be used directly for inference or for reproducing the ensemble-based results presented in the study.
```
trained_models/
├── model1.h5
├── model2.h5
...
├── model12.h5
```

## Optional data augmentation

Data augmentation is separated from the training/test split. The notebook first creates the split, then optionally applies mirror-based augmentation to the training subset only. To disable this step, edit `src/config.py`:

```python
USE_DATA_AUGMENTATION = True
```

The test subset is never augmented.

### Optional pre-processing

The pre-processing steps can be activated independently in `src/config.py`:

```python
USE_CLAHE = True
USE_HOG = True
```

- `USE_CLAHE` controls Contrast Limited Adaptive Histogram Equalisation during data loading.
- `USE_HOG` controls HOG-based rotation of anisotropic image patches before the training/test split.

## Optional MobileNetV2 ImageNet transfer-learning test

The repository also includes an optional transfer-learning baseline requested during peer review. It uses `MobileNetV2` with `weights="imagenet"`, `include_top=False` and a custom classification head for the 12 rooftop material classes. The original 64 x 64 patches are not resampled on disk; they are resized to 224 x 224 inside the model before MobileNetV2 pre-processing.

The experiment is controlled from `src/config.py`:

```python
RUN_MOBILENETV2_TEST = True
MOBILENETV2_INPUT_SIZE = (224, 224)
MOBILENETV2_BATCH_SIZE = 32
MOBILENETV2_EPOCHS = 50
MOBILENETV2_FINE_TUNE = False
```

By default, the ImageNet backbone is frozen and only the new classification head is trained. Optional fine-tuning of the upper MobileNetV2 layers can be enabled with `MOBILENETV2_FINE_TUNE = True`.

The ImageNet weights are downloaded automatically by TensorFlow/Keras on the first run if they are not already cached locally.

## Optional weighted categorical cross-entropy

A reviewer-requested class-imbalance experiment is included. Class weights are computed from the training labels only, after the training/test split and after optional training-set augmentation. The test subset is never used to compute class weights.

The experiment is controlled from `src/config.py`:

```python
USE_WEIGHTED_CATEGORICAL_CROSSENTROPY = True
```

When enabled, the CNN ensemble (and the optional MobileNetV2 baseline) are compiled with a weighted categorical cross-entropy loss. When disabled, standard categorical cross-entropy is used.

## Adjustable ensemble size

The framework supports ensemble learning based on multiple independently trained CNNs. The number of models included in the ensemble can be freely adjusted in `src/config.py`, allowing users to investigate the influence of ensemble size on classification performance and model stability, while balancing computational cost and training time.

```python
N_MODELS = 12
