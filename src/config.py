"""
Configuration module for Rice Crop Disease Detection.
Defines paths, hyperparameters, and feature extraction settings.
Based on Research.pdf and ML ALL.pdf.
"""

import os

# Root directory of the MLP2 project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths in MLP2
DATASETS = {
    'DS1': {
        'name': 'Rice Leaf Disease Images (DS1)',
        'path': os.path.join(BASE_DIR, 'Rice Leaf Disease Images'),
        'description': '4 classes: Bacterialblight, Blast, Brownspot, Tungro (5,932 images)'
    },
    'DS2': {
        'name': 'Rice Diseases Dataset (DS2)',
        'train_path': os.path.join(BASE_DIR, 'Dataset', 'Full_Train_Data'),
        'test_path': os.path.join(BASE_DIR, 'Dataset', 'Test_Data_SVM'),
        'description': '5 classes: blight, blast, Brownspot, tungro, sheath_blight (2,800 images)'
    },
    'DS3': {
        'name': 'Rice Leaf Diseases Dataset (DS3)',
        'path': os.path.join(BASE_DIR, 'rice leaf diseases dataset'),
        'description': '3 classes: Bacterialblight, Brownspot, Leafsmut (4,684 images)'
    }
}

# Image Preprocessing Hyperparameters (Research.pdf Eq. 2, Eq. 7-8)
IMAGE_SIZE = (224, 224)    # Standard size 224x224 (Eq. 2)
LOG_C = 2.0                # Logarithmic enhancement scaling factor c=2 (Eq. 8)

# Augmentation Parameters (Research.pdf Eq. 3-6)
AUGMENT_TRANSLATION_RANGE = (-5, 15)  # Translation in pixels along X/Y (Eq. 4)
AUGMENT_ROTATION_RANGE = (-10, 10)    # Rotation in degrees (Eq. 6)

# Feature Extraction Hyperparameters (Research.pdf Eq. 9-13)
ML_LEVELS = 3              # Multi-Level spatial blocks grid (L x L), defaults to 3 (9 blocks)
LBP_RADIUS = 1             # LBP radius
LBP_POINTS = 8             # LBP neighboring points (Eq. 11)
CORRELOGRAM_DISTANCES = [1, 3, 5, 7]  # Spatial distances D for Color Correlogram (Eq. 9)
COLOR_BINS_PER_CHANNEL = 4 # 4 bins per channel -> 4x4x4 = 64 color quantization

# Dataset Partitioning Hyperparameters (70% Train, 10% Validation, 20% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Storage paths
CACHE_DIR = os.path.join(BASE_DIR, 'features_cache')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
