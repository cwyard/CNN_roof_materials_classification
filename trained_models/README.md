# Model configuration and associated parameters to modify in /src/config.py
- CLAHE: enabled -> USE_CLAHE = True
- HOG rotation: enabled --> USE_HOG = True
- Data augmentation: enabled --> USE_DATA_AUGMENTATION = True
- Weighted loss: disabled --> USE_WEIGHTED_CATEGORICAL_CROSSENTROPY = False
- Ensemble size: 12 CNNs --> N_MODELS = 12
- Optional transfer learning : disabled --> RUN_MOBILENETV2_TEST = False


